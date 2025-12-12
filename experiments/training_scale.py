"""
Experiment: Minimum training scale to differentiate architecture quality.

Research question: How many training steps are needed before good and bad
architectures show clearly different loss trajectories?

This informs:
- How long to train during validation
- Whether we can do quick filtering before expensive training
- Cost/benefit tradeoffs for architecture search

Methodology:
1. Train a known-good architecture (basic Transformer) at various step counts
2. Train a known-bad architecture (mismatched components) at same scales
3. Track loss curves and look for divergence point
4. Repeat with different seeds for statistical significance
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import math
import time
from dataclasses import dataclass, field
from typing import Optional

# Check for torch
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch required. Install with: pip install torch")
    sys.exit(1)

from arcfusion import ArcFusionDB
from arcfusion.validator import (
    ValidationPipeline,
    ModelConfig,
    TrainingConfig,
    SyntheticDataset,
    DEFAULT_VOCAB_SIZE,
    DEFAULT_SEQ_LEN,
)
from arcfusion.codegen import GeneratedCode


# Training scales to test (number of steps)
TRAINING_SCALES = [100, 250, 500, 1000, 2000]

# Number of random seeds for statistical significance
NUM_SEEDS = 3

# Model configuration (small for fast iteration)
MODEL_CONFIG = ModelConfig(
    d_model=128,
    vocab_size=1000,
    max_seq_len=64,
    n_heads=4,
    n_layers=2,
)


@dataclass
class ExperimentResult:
    """Result from a single training run."""
    architecture: str  # "good" or "bad"
    seed: int
    max_steps: int
    loss_history: list = field(default_factory=list)
    final_loss: float = float('inf')
    training_time: float = 0.0
    num_parameters: int = 0
    error: Optional[str] = None


# Good architecture: Standard Transformer block
GOOD_ARCHITECTURE_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GoodTransformer(nn.Module):
    """Standard Transformer: attention + FFN with residual connections."""

    def __init__(self, d_model=128, vocab_size=1000, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        return x
'''


# Bad architecture: Random mixing without proper structure
BAD_ARCHITECTURE_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BadArchitecture(nn.Module):
    """Poorly designed: no residuals, wrong normalization order, mismatched dims."""

    def __init__(self, d_model=128, vocab_size=1000, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Bad: no residual connections, norm after instead of before
        self.layers = nn.ModuleList([
            BadBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BadBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        # Bad: Linear instead of attention - loses sequence modeling
        self.proj = nn.Linear(d_model, d_model)
        # Bad: tiny FFN bottleneck
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),  # Bad: tanh instead of GELU/ReLU
            nn.Linear(d_model // 4, d_model),
        )
        # Bad: norm after instead of before
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # No residual connections - gradient flow problems
        x = self.proj(x)
        x = self.ffn(x)
        x = self.norm(x)
        return x
'''


def create_model_with_head(code: str, model_name: str, config: ModelConfig, device: str):
    """Create model from code string with embedding and LM head."""
    namespace = {'torch': torch, 'nn': nn, 'F': torch.nn.functional, 'math': math}
    exec(code, namespace)

    model_class = namespace[model_name]
    model = model_class(
        d_model=config.d_model,
        vocab_size=config.vocab_size,
    ).to(device)

    # Add embedding and LM head
    embedding = nn.Embedding(config.vocab_size, config.d_model).to(device)
    lm_head = nn.Linear(config.d_model, config.vocab_size).to(device)

    return model, embedding, lm_head


def train_with_logging(
    model: nn.Module,
    embedding: nn.Module,
    lm_head: nn.Module,
    config: ModelConfig,
    max_steps: int,
    seed: int,
    device: str = 'cpu',
    log_interval: int = 25,
) -> tuple[list, float, float]:
    """
    Train model and return loss history.

    Returns: (loss_history, final_loss, training_time)
    """
    torch.manual_seed(seed)

    # Create dataset
    dataset = SyntheticDataset(config.vocab_size, config.max_seq_len)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, drop_last=True
    )

    # Optimizer for all parameters
    all_params = list(model.parameters()) + list(embedding.parameters()) + list(lm_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    loss_history = []
    start_time = time.time()

    data_iter = iter(dataloader)
    step = 0
    running_loss = 0.0

    while step < max_steps:
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inputs, targets = next(data_iter)

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward
        embedded = embedding(inputs)
        hidden = model(embedded)
        logits = lm_head(hidden)

        loss = criterion(logits.view(-1, config.vocab_size), targets.view(-1))

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        loss_val = loss.item()
        running_loss += loss_val
        step += 1

        # Log at intervals
        if step % log_interval == 0:
            avg_loss = running_loss / step
            loss_history.append({'step': step, 'loss': avg_loss})

    training_time = time.time() - start_time
    final_loss = running_loss / max(step, 1)

    return loss_history, final_loss, training_time


def run_experiment(
    architecture: str,
    code: str,
    model_name: str,
    max_steps: int,
    seed: int,
    device: str = 'cpu',
) -> ExperimentResult:
    """Run a single experiment."""
    result = ExperimentResult(
        architecture=architecture,
        seed=seed,
        max_steps=max_steps,
    )

    try:
        model, embedding, lm_head = create_model_with_head(
            code, model_name, MODEL_CONFIG, device
        )
        result.num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        loss_history, final_loss, training_time = train_with_logging(
            model, embedding, lm_head, MODEL_CONFIG, max_steps, seed, device
        )

        result.loss_history = loss_history
        result.final_loss = final_loss
        result.training_time = training_time

    except Exception as e:
        result.error = str(e)

    return result


def analyze_divergence(good_results: list[ExperimentResult], bad_results: list[ExperimentResult]):
    """Analyze when good and bad architectures diverge."""
    print("\n" + "="*60)
    print("DIVERGENCE ANALYSIS")
    print("="*60)

    # Group by step count
    for max_steps in TRAINING_SCALES:
        good_at_scale = [r for r in good_results if r.max_steps == max_steps]
        bad_at_scale = [r for r in bad_results if r.max_steps == max_steps]

        if not good_at_scale or not bad_at_scale:
            continue

        # Average final losses
        good_avg = sum(r.final_loss for r in good_at_scale) / len(good_at_scale)
        bad_avg = sum(r.final_loss for r in bad_at_scale) / len(bad_at_scale)

        # Standard deviation
        good_std = math.sqrt(sum((r.final_loss - good_avg)**2 for r in good_at_scale) / len(good_at_scale))
        bad_std = math.sqrt(sum((r.final_loss - bad_avg)**2 for r in bad_at_scale) / len(bad_at_scale))

        # Difference in standard deviations
        diff = bad_avg - good_avg
        separation = diff / (good_std + bad_std + 1e-6)  # How many stddevs apart

        print(f"\nAt {max_steps} steps:")
        print(f"  Good: loss={good_avg:.4f} (±{good_std:.4f})")
        print(f"  Bad:  loss={bad_avg:.4f} (±{bad_std:.4f})")
        print(f"  Difference: {diff:.4f} ({separation:.1f}σ separation)")

        if separation > 2.0:
            print(f"  ✓ CLEAR DIVERGENCE (>{2.0:.1f}σ)")
        elif separation > 1.0:
            print(f"  ~ Moderate separation")
        else:
            print(f"  ✗ No clear divergence yet")


def main():
    """Run the training scale experiment."""
    print("="*60)
    print("TRAINING SCALE EXPERIMENT")
    print("="*60)
    print(f"Scales: {TRAINING_SCALES} steps")
    print(f"Seeds: {NUM_SEEDS}")
    print(f"Model: d_model={MODEL_CONFIG.d_model}, vocab={MODEL_CONFIG.vocab_size}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    all_results = []

    # Run good architecture at all scales
    print("\n" + "-"*40)
    print("Testing GOOD architecture (Transformer)")
    print("-"*40)

    for max_steps in TRAINING_SCALES:
        for seed in range(NUM_SEEDS):
            print(f"  Running: steps={max_steps}, seed={seed}...", end=" ", flush=True)
            result = run_experiment(
                architecture="good",
                code=GOOD_ARCHITECTURE_CODE,
                model_name="GoodTransformer",
                max_steps=max_steps,
                seed=seed,
                device=device,
            )
            all_results.append(result)
            if result.error:
                print(f"ERROR: {result.error}")
            else:
                print(f"loss={result.final_loss:.4f}, time={result.training_time:.1f}s")

    # Run bad architecture at all scales
    print("\n" + "-"*40)
    print("Testing BAD architecture (no attention/residuals)")
    print("-"*40)

    for max_steps in TRAINING_SCALES:
        for seed in range(NUM_SEEDS):
            print(f"  Running: steps={max_steps}, seed={seed}...", end=" ", flush=True)
            result = run_experiment(
                architecture="bad",
                code=BAD_ARCHITECTURE_CODE,
                model_name="BadArchitecture",
                max_steps=max_steps,
                seed=seed,
                device=device,
            )
            all_results.append(result)
            if result.error:
                print(f"ERROR: {result.error}")
            else:
                print(f"loss={result.final_loss:.4f}, time={result.training_time:.1f}s")

    # Analyze results
    good_results = [r for r in all_results if r.architecture == "good" and not r.error]
    bad_results = [r for r in all_results if r.architecture == "bad" and not r.error]

    analyze_divergence(good_results, bad_results)

    # Summary recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    # Find first clear divergence point
    for max_steps in TRAINING_SCALES:
        good_at_scale = [r for r in good_results if r.max_steps == max_steps]
        bad_at_scale = [r for r in bad_results if r.max_steps == max_steps]

        if good_at_scale and bad_at_scale:
            good_avg = sum(r.final_loss for r in good_at_scale) / len(good_at_scale)
            bad_avg = sum(r.final_loss for r in bad_at_scale) / len(bad_at_scale)
            good_std = math.sqrt(sum((r.final_loss - good_avg)**2 for r in good_at_scale) / len(good_at_scale))
            bad_std = math.sqrt(sum((r.final_loss - bad_avg)**2 for r in bad_at_scale) / len(bad_at_scale))
            separation = (bad_avg - good_avg) / (good_std + bad_std + 1e-6)

            if separation > 2.0:
                print(f"\n✓ Minimum training for clear differentiation: {max_steps} steps")
                print(f"  - Use this for quick validation passes")
                print(f"  - Good architectures should show loss < {good_avg + good_std:.4f}")
                break
    else:
        print("\n⚠ No clear divergence found - may need more steps or different bad architecture")

    # Save results to JSON
    output_file = os.path.join(os.path.dirname(__file__), 'training_scale_results.json')
    results_data = [
        {
            'architecture': r.architecture,
            'seed': r.seed,
            'max_steps': r.max_steps,
            'final_loss': r.final_loss,
            'training_time': r.training_time,
            'num_parameters': r.num_parameters,
            'loss_history': r.loss_history,
            'error': r.error,
        }
        for r in all_results
    ]
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
