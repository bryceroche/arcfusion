"""
Cloud Training Integration - Run validation on cloud GPUs via Modal.

Enables training dreamed architectures at scale without local GPU.
Uses Modal for serverless GPU compute with pay-per-second billing.

Usage:
    from arcfusion.cloud import CloudTrainer

    trainer = CloudTrainer()
    result = trainer.train(generated_code, model_name, config)
"""

from dataclasses import dataclass, field
from typing import Optional

# Check for Modal availability
try:
    import modal
    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False
    modal = None

# Modal app configuration
APP_NAME = "arcfusion-trainer"
GPU_TYPE = "T4"  # Good balance of cost/performance for small models
TIMEOUT_SECONDS = 600  # 10 minute max per training run
IMAGE_PYTHON_VERSION = "3.11"

# Training defaults for cloud runs
# Note: Research shows 2000+ steps needed to differentiate architecture quality
# See experiments/training_scale.py and ideas.md for details
CLOUD_DEFAULT_STEPS = 2000
CLOUD_DEFAULT_BATCH_SIZE = 16
CLOUD_DEFAULT_D_MODEL = 256
CLOUD_DEFAULT_VOCAB_SIZE = 8000


@dataclass
class CloudConfig:
    """Configuration for cloud training runs."""
    gpu_type: str = GPU_TYPE
    timeout_seconds: int = TIMEOUT_SECONDS
    max_steps: int = CLOUD_DEFAULT_STEPS
    batch_size: int = CLOUD_DEFAULT_BATCH_SIZE
    d_model: int = CLOUD_DEFAULT_D_MODEL
    vocab_size: int = CLOUD_DEFAULT_VOCAB_SIZE
    learning_rate: float = 1e-4
    eval_interval: int = 50


@dataclass
class CloudResult:
    """Result from cloud training run."""
    success: bool
    # Training metrics
    final_loss: float = float('inf')
    perplexity: float = float('inf')
    steps_completed: int = 0
    training_time_seconds: float = 0.0
    # Model info
    num_parameters: int = 0
    # Errors
    error: Optional[str] = None
    # Loss history for analysis
    loss_history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'final_loss': self.final_loss,
            'perplexity': self.perplexity,
            'steps_completed': self.steps_completed,
            'training_time_seconds': self.training_time_seconds,
            'num_parameters': self.num_parameters,
            'error': self.error,
            'loss_history': self.loss_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CloudResult':
        """Create from dictionary."""
        return cls(**data)


def _create_modal_app():
    """Create Modal app with PyTorch GPU image."""
    if not HAS_MODAL:
        raise ImportError("Modal not installed. Install with: pip install modal")

    app = modal.App(APP_NAME)

    # Create image with PyTorch and dependencies
    image = (
        modal.Image.debian_slim(python_version=IMAGE_PYTHON_VERSION)
        .pip_install(
            "torch>=2.0",
            "numpy",
        )
    )

    return app, image


# Define the remote training function
if HAS_MODAL:
    app, image = _create_modal_app()

    @app.function(
        image=image,
        gpu=GPU_TYPE,
        timeout=TIMEOUT_SECONDS,
    )
    def _train_remote_modal(
        code: str,
        model_name: str,
        config_dict: dict,
    ) -> dict:
        """
        Remote training function that runs on Modal GPU.

        Args:
            code: Generated PyTorch model code
            model_name: Name of the model class to instantiate
            config_dict: Training configuration as dict

        Returns:
            CloudResult as dict
        """
        import time
        import math
        import torch
        import torch.nn as nn

        result = {
            'success': False,
            'final_loss': float('inf'),
            'perplexity': float('inf'),
            'steps_completed': 0,
            'training_time_seconds': 0.0,
            'num_parameters': 0,
            'error': None,
            'loss_history': [],
        }

        start_time = time.time()

        try:
            # Execute the generated code
            namespace = {}
            exec(code, namespace)

            # Get the model class
            model_class = namespace.get(model_name)
            if model_class is None:
                result['error'] = f"Model class '{model_name}' not found"
                return result

            # Instantiate model
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model_class(
                d_model=config_dict['d_model'],
                vocab_size=config_dict['vocab_size'],
            ).to(device)

            # Count parameters
            result['num_parameters'] = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            # Create synthetic data
            vocab_size = config_dict['vocab_size']
            batch_size = config_dict['batch_size']
            seq_len = 64

            # Training setup
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config_dict['learning_rate']
            )
            criterion = nn.CrossEntropyLoss()

            # Check if model expects token ids or embeddings
            needs_embedding = not hasattr(model, 'embedding')
            if needs_embedding:
                embedding = nn.Embedding(vocab_size, config_dict['d_model']).to(device)
                lm_head = nn.Linear(config_dict['d_model'], vocab_size).to(device)
                # Add embedding params to optimizer
                optimizer.add_param_group({'params': embedding.parameters()})
                optimizer.add_param_group({'params': lm_head.parameters()})

            # Training loop
            model.train()
            max_steps = config_dict['max_steps']
            eval_interval = config_dict.get('eval_interval', 50)

            for step in range(max_steps):
                # Generate random batch
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]

                # Forward pass
                if needs_embedding:
                    x = embedding(inputs)
                    x = model(x)
                    logits = lm_head(x)
                else:
                    logits = model(inputs)

                # Compute loss
                loss = criterion(
                    logits.view(-1, vocab_size),
                    targets.view(-1)
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # Record loss
                loss_val = loss.item()
                if step % eval_interval == 0:
                    result['loss_history'].append({
                        'step': step,
                        'loss': loss_val,
                    })

                result['steps_completed'] = step + 1
                result['final_loss'] = loss_val

            # Compute final perplexity
            if result['final_loss'] < 20:
                result['perplexity'] = math.exp(result['final_loss'])
            else:
                result['perplexity'] = float('inf')

            result['success'] = True

        except Exception as e:
            result['error'] = str(e)

        result['training_time_seconds'] = time.time() - start_time
        return result


class CloudTrainer:
    """
    Cloud training interface for ArcFusion.

    Uses Modal to run training on serverless GPUs.
    """

    def __init__(self, config: Optional[CloudConfig] = None):
        if not HAS_MODAL:
            raise ImportError(
                "Modal not installed. Install with: pip install 'arcfusion[cloud]'"
            )
        self.config = config or CloudConfig()

    def train(
        self,
        code: str,
        model_name: str,
        config: Optional[CloudConfig] = None,
        verbose: bool = True,
    ) -> CloudResult:
        """
        Train a model on cloud GPU.

        Args:
            code: Generated PyTorch model code
            model_name: Name of the model class in the code
            config: Optional override for training config
            verbose: Print progress

        Returns:
            CloudResult with training metrics
        """
        cfg = config or self.config

        if verbose:
            print(f"Starting cloud training on Modal ({cfg.gpu_type})...")
            print(f"  Model: {model_name}")
            print(f"  Steps: {cfg.max_steps}")
            print(f"  Batch size: {cfg.batch_size}")

        # Call remote function
        with app.run():
            result_dict = _train_remote_modal.remote(
                code=code,
                model_name=model_name,
                config_dict={
                    'd_model': cfg.d_model,
                    'vocab_size': cfg.vocab_size,
                    'batch_size': cfg.batch_size,
                    'learning_rate': cfg.learning_rate,
                    'max_steps': cfg.max_steps,
                    'eval_interval': cfg.eval_interval,
                },
            )

        result = CloudResult.from_dict(result_dict)

        if verbose:
            if result.success:
                print("\nTraining complete!")
                print(f"  Final loss: {result.final_loss:.4f}")
                print(f"  Perplexity: {result.perplexity:.2f}")
                print(f"  Parameters: {result.num_parameters:,}")
                print(f"  Time: {result.training_time_seconds:.1f}s")
            else:
                print(f"\nTraining failed: {result.error}")

        return result

    def train_recipe(
        self,
        recipe: 'Recipe',
        db: 'ArcFusionDB',
        config: Optional[CloudConfig] = None,
        verbose: bool = True,
    ) -> CloudResult:
        """
        Train a recipe on cloud GPU.

        Convenience method that generates code from recipe first.

        Args:
            recipe: Recipe from Composer
            db: Database for looking up components
            config: Optional override for training config
            verbose: Print progress

        Returns:
            CloudResult with training metrics
        """
        from .codegen import CodeGenerator
        from .composer import EngineComposer

        composer = EngineComposer(db)
        codegen = CodeGenerator(db)

        # Get components and generate code
        components = composer.recipe_to_components(recipe)
        generated = codegen.generate(
            components=components,
            name=recipe.name,
        )

        # Validate syntax locally first
        valid, error = generated.validate_syntax()
        if not valid:
            return CloudResult(
                success=False,
                error=f"Code generation failed: {error}",
            )

        return self.train(
            code=generated.code,
            model_name=generated.name,
            config=config,
            verbose=verbose,
        )
