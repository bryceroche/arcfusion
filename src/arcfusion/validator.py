"""
Auto-Validation Pipeline - Build, train, and benchmark dreamed architectures.

Takes generated code from CodeGenerator and:
1. Compiles it into a runnable PyTorch model
2. Trains on small datasets (WikiText-2, TinyStories, etc.)
3. Runs benchmarks (perplexity, accuracy)
4. Updates component scores based on real performance
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    DataLoader = None
    Dataset = object  # Fallback base class

from .db import ArcFusionDB, BenchmarkResult, Component
from .codegen import GeneratedCode

# Validation constants
DEFAULT_D_MODEL = 128  # Small for fast iteration
DEFAULT_VOCAB_SIZE = 1000  # Tiny vocab for testing
DEFAULT_SEQ_LEN = 64
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_STEPS = 100
DEFAULT_LEARNING_RATE = 1e-4

# Perplexity calculation constants
MAX_LOSS_FOR_PERPLEXITY = 20.0  # Clamp loss to avoid overflow in exp()
MAX_PERPLEXITY_FOR_SCORING = 1000.0  # Scale for converting ppl to 0-1 score

# Evaluation constants
DEFAULT_EVAL_BATCHES = 10  # Number of batches for evaluation


def _create_embedding_and_head(
    model: 'nn.Module',
    device: str,
    vocab_size: Optional[int] = None
) -> tuple:
    """
    Create embedding and LM head layers for models that expect embedded input.

    Args:
        model: The model (used to get d_model)
        device: Device to place layers on
        vocab_size: Vocabulary size (uses model's vocab_size or default if not provided)

    Returns:
        (embedding, lm_head) tuple of nn.Module layers
    """
    # Determine vocab_size: explicit param > model attribute > default
    if vocab_size is None:
        vocab_size = getattr(model, 'vocab_size', DEFAULT_VOCAB_SIZE)
    d_model = model.d_model if hasattr(model, 'd_model') else DEFAULT_D_MODEL
    embedding = nn.Embedding(vocab_size, d_model).to(device)
    lm_head = nn.Linear(d_model, vocab_size).to(device)
    return embedding, lm_head


@dataclass
class ModelConfig:
    """Configuration for model building."""
    d_model: int = DEFAULT_D_MODEL
    vocab_size: int = DEFAULT_VOCAB_SIZE
    max_seq_len: int = DEFAULT_SEQ_LEN
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = DEFAULT_BATCH_SIZE
    max_steps: int = DEFAULT_MAX_STEPS
    learning_rate: float = DEFAULT_LEARNING_RATE
    max_time_seconds: float = 300.0  # 5 minute timeout
    eval_interval: int = 50
    device: str = "cpu"


@dataclass
class ValidationResult:
    """Result of validating a dreamed architecture."""
    success: bool
    model_name: str
    num_parameters: int = 0
    build_error: Optional[str] = None
    train_error: Optional[str] = None
    final_loss: float = float('inf')
    perplexity: float = float('inf')
    training_steps: int = 0
    training_time_seconds: float = 0.0
    benchmarks: dict = field(default_factory=dict)

    @property
    def is_trainable(self) -> bool:
        """Model built and trained without errors."""
        return self.success and self.build_error is None and self.train_error is None


class ModelBuilder:
    """Compile generated code into a runnable PyTorch model."""

    def __init__(self, config: Optional[ModelConfig] = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch required. Install with: pip install torch")
        self.config = config or ModelConfig()

    def build(self, generated: GeneratedCode) -> tuple[Optional[nn.Module], Optional[str]]:
        """
        Build a PyTorch model from generated code.

        Args:
            generated: GeneratedCode from CodeGenerator

        Returns:
            (model, error) - model is None if build failed
        """
        # First validate syntax
        valid, error = generated.validate_syntax()
        if not valid:
            return None, f"Syntax error: {error}"

        # Create a namespace to exec the code into
        namespace = {
            'torch': torch,
            'nn': nn,
            'F': F,
            'math': math,
        }

        try:
            # Execute the generated code
            exec(generated.code, namespace)

            # Find the main architecture class
            model_class = namespace.get(generated.name)
            if model_class is None:
                return None, f"Class '{generated.name}' not found in generated code"

            # Instantiate the model
            model = model_class(
                d_model=self.config.d_model,
                vocab_size=self.config.vocab_size,
            )

            return model, None

        except Exception as e:
            return None, f"Build error: {str(e)}"

    def count_parameters(self, model: nn.Module) -> int:
        """Count trainable parameters in a model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SyntheticDataset(Dataset):
    """Simple synthetic dataset for testing - random token sequences."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        # Pre-generate data for consistency
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple:
        # Return (input, target) where target is input shifted by 1
        seq = self.data[idx]
        return seq[:-1], seq[1:]


class TrainingHarness:
    """Train models on small datasets."""

    def __init__(self, config: Optional[TrainingConfig] = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch required. Install with: pip install torch")
        self.config = config or TrainingConfig()

    def create_dataloader(
        self,
        vocab_size: int,
        seq_len: int,
        num_samples: int = 1000
    ) -> DataLoader:
        """Create a simple dataloader for training."""
        dataset = SyntheticDataset(vocab_size, seq_len, num_samples)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )

    def train(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        verbose: bool = False,
        vocab_size: Optional[int] = None
    ) -> tuple[float, int, Optional[str]]:
        """
        Train a model for a limited number of steps.

        Args:
            model: PyTorch model to train
            dataloader: DataLoader with training data
            verbose: Print progress
            vocab_size: Vocabulary size for embedding/head layers

        Returns:
            (final_loss, steps_completed, error)
        """
        device = self.config.device
        model = model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate
        )

        start_time = time.time()
        step = 0
        running_loss = 0.0
        final_loss = float('inf')
        embedding = None
        lm_head = None

        try:
            data_iter = iter(dataloader)

            while step < self.config.max_steps:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > self.config.max_time_seconds:
                    if verbose:
                        print(f"  Timeout after {elapsed:.1f}s")
                    break

                # Get batch (restart iterator if exhausted)
                try:
                    inputs, targets = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    inputs, targets = next(data_iter)

                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                # Create embedding/head on first batch
                if embedding is None:
                    embedding, lm_head = _create_embedding_and_head(model, device, vocab_size)

                embedded = embedding(inputs)
                outputs = model(embedded)

                # Check if model already outputs vocab-sized logits
                effective_vocab_size = vocab_size or getattr(model, 'vocab_size', lm_head.out_features)
                if outputs.size(-1) >= effective_vocab_size:
                    logits = outputs  # Model has built-in LM head
                else:
                    logits = lm_head(outputs)

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )

                # Backward pass
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                step += 1

                if verbose and step % self.config.eval_interval == 0:
                    avg_loss = running_loss / step
                    print(f"  Step {step}: loss={avg_loss:.4f}, ppl={math.exp(min(avg_loss, MAX_LOSS_FOR_PERPLEXITY)):.2f}")

            final_loss = running_loss / max(step, 1)
            return final_loss, step, None

        except Exception as e:
            return float('inf'), step, f"Training error: {str(e)}"


class BenchmarkRunner:
    """Run benchmarks on trained models."""

    def __init__(self, db: Optional[ArcFusionDB] = None):
        self.db = db

    def compute_perplexity(self, loss: float) -> float:
        """Compute perplexity from cross-entropy loss."""
        # Clamp to avoid overflow
        return math.exp(min(loss, MAX_LOSS_FOR_PERPLEXITY))

    def run_benchmarks(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        config: TrainingConfig,
        vocab_size: Optional[int] = None
    ) -> dict:
        """
        Run evaluation benchmarks on a model.

        Args:
            model: PyTorch model to evaluate
            dataloader: DataLoader with evaluation data
            config: Training configuration
            vocab_size: Vocabulary size for embedding/head layers

        Returns:
            Dict of benchmark_name -> score
        """
        if not HAS_TORCH:
            return {}

        device = config.device
        model = model.to(device)
        model.eval()

        total_loss = 0.0
        num_batches = 0
        embedding = None
        lm_head = None

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Create embedding/head on first batch
                if embedding is None:
                    embedding, lm_head = _create_embedding_and_head(model, device, vocab_size)

                embedded = embedding(inputs)
                outputs = model(embedded)

                # Check if model already outputs vocab-sized logits
                effective_vocab_size = vocab_size or getattr(model, 'vocab_size', lm_head.out_features)
                if outputs.size(-1) >= effective_vocab_size:
                    logits = outputs  # Model has built-in LM head
                else:
                    logits = lm_head(outputs)

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )

                total_loss += loss.item()
                num_batches += 1

                # Limit eval batches
                if num_batches >= DEFAULT_EVAL_BATCHES:
                    break

        avg_loss = total_loss / max(num_batches, 1)
        perplexity = self.compute_perplexity(avg_loss)

        return {
            'eval_loss': avg_loss,
            'perplexity': perplexity,
        }

    def store_result(
        self,
        engine_id: str,
        benchmark_name: str,
        score: float,
        parameters: Optional[dict] = None,
        notes: str = ""
    ) -> Optional[str]:
        """Store benchmark result in database."""
        if not self.db:
            return None

        result = BenchmarkResult(
            engine_id=engine_id,
            benchmark_name=benchmark_name,
            score=score,
            parameters=parameters or {},
            notes=notes
        )
        return self.db.add_benchmark(result)


class ValidationPipeline:
    """
    Full validation pipeline: build → train → benchmark → feedback.
    """

    def __init__(
        self,
        db: Optional[ArcFusionDB] = None,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.db = db
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()

        self.builder = ModelBuilder(self.model_config)
        self.harness = TrainingHarness(self.training_config)
        self.benchmarker = BenchmarkRunner(db)

    def validate(
        self,
        generated: GeneratedCode,
        verbose: bool = True
    ) -> ValidationResult:
        """
        Run full validation pipeline on generated code.

        Args:
            generated: GeneratedCode from CodeGenerator
            verbose: Print progress

        Returns:
            ValidationResult with all metrics
        """
        result = ValidationResult(
            success=False,
            model_name=generated.name,
        )

        # Step 1: Build model
        if verbose:
            print(f"Building model: {generated.name}...")

        model, build_error = self.builder.build(generated)
        if build_error:
            result.build_error = build_error
            if verbose:
                print(f"  [FAILED] {build_error}")
            return result

        result.num_parameters = self.builder.count_parameters(model)
        if verbose:
            print(f"  Parameters: {result.num_parameters:,}")

        # Step 2: Train model
        if verbose:
            print(f"Training for {self.training_config.max_steps} steps...")

        dataloader = self.harness.create_dataloader(
            vocab_size=self.model_config.vocab_size,
            seq_len=self.model_config.max_seq_len,
        )

        start_time = time.time()
        final_loss, steps, train_error = self.harness.train(
            model, dataloader, verbose=verbose, vocab_size=self.model_config.vocab_size
        )
        training_time = time.time() - start_time

        if train_error:
            result.train_error = train_error
            if verbose:
                print(f"  [FAILED] {train_error}")
            return result

        result.final_loss = final_loss
        result.perplexity = self.benchmarker.compute_perplexity(final_loss)
        result.training_steps = steps
        result.training_time_seconds = training_time

        if verbose:
            print(f"  Final loss: {final_loss:.4f}")
            print(f"  Perplexity: {result.perplexity:.2f}")
            print(f"  Time: {training_time:.1f}s")

        # Step 3: Run benchmarks
        if verbose:
            print("Running benchmarks...")

        benchmarks = self.benchmarker.run_benchmarks(
            model, dataloader, self.training_config, vocab_size=self.model_config.vocab_size
        )
        result.benchmarks = benchmarks

        if verbose:
            for name, score in benchmarks.items():
                print(f"  {name}: {score:.4f}")

        result.success = True
        return result

    def validate_and_store(
        self,
        generated: GeneratedCode,
        engine_id: Optional[str] = None,
        verbose: bool = True
    ) -> ValidationResult:
        """
        Validate and store results in database.

        Args:
            generated: GeneratedCode to validate
            engine_id: Optional engine ID to associate results with
            verbose: Print progress

        Returns:
            ValidationResult
        """
        result = self.validate(generated, verbose=verbose)

        if self.db and engine_id and result.success:
            # Store benchmark results
            for bench_name, score in result.benchmarks.items():
                self.benchmarker.store_result(
                    engine_id=engine_id,
                    benchmark_name=bench_name,
                    score=score,
                    parameters={
                        'd_model': self.model_config.d_model,
                        'vocab_size': self.model_config.vocab_size,
                        'max_steps': self.training_config.max_steps,
                    },
                    notes=f"Auto-validated: {generated.name}"
                )

            if verbose:
                print(f"Stored {len(result.benchmarks)} benchmark results")

        return result

    def update_component_scores(
        self,
        components: list[Component],
        result: ValidationResult,
        weight: float = 0.1
    ) -> None:
        """
        Update component usefulness_score based on validation results.

        Uses exponential moving average to blend old and new scores.
        """
        if not self.db or not result.success:
            return

        # Convert perplexity to a 0-1 score (lower perplexity = higher score)
        # Perplexity of ~1 is perfect, >100 is bad
        # Use log scale: score = 1 - (log(ppl) / log(max_ppl))
        ppl_score = max(0.0, 1.0 - math.log(result.perplexity + 1) / math.log(MAX_PERPLEXITY_FOR_SCORING))

        for comp in components:
            # Blend old score with new observation
            old_score = comp.usefulness_score
            new_score = old_score * (1 - weight) + ppl_score * weight

            # Update in database
            self.db.conn.execute(
                "UPDATE components SET usefulness_score = ? WHERE component_id = ?",
                (new_score, comp.component_id)
            )

        self.db.conn.commit()
