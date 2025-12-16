#!/usr/bin/env python3
"""
Fast Mamba: Parallel Scan Implementation

Replaces the O(L) sequential loop with O(log L) parallel scan.
This should dramatically speed up Mamba training.

Run with: PYTHONUNBUFFERED=1 .venv-modal/bin/python scripts/run_mamba_fast.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "arcfusion"))

from cloud_train_fair import CONFIG, app, train_model, save_result_to_db, generate_auto_insight
from db import ArcFusionDB


# Fast Mamba using parallel scan
MAMBA_FAST_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F


def parallel_scan(A, B_x):
    """Parallel associative scan for SSM.

    Computes h_t = A_t * h_{t-1} + B_x_t for all t in parallel.

    Args:
        A: (B, L, D, N) - state transition coefficients (already exp'd)
        B_x: (B, L, D, N) - input contribution (B_bar * x)

    Returns:
        h: (B, L, D, N) - all hidden states

    The key insight is that the SSM recurrence h_t = A*h_{t-1} + b is associative
    when viewed as (A, b) tuples with composition (A1, b1) ∘ (A2, b2) = (A1*A2, A1*b2 + b1)
    """
    B, L, D, N = A.shape

    # For efficiency, use cumulative product approach for diagonal A
    # h_t = sum_{s=0}^{t} (prod_{k=s+1}^{t} A_k) * B_x_s

    # Compute cumulative product of A from right to left
    # log_A for numerical stability
    log_A = torch.log(A.clamp(min=1e-6))
    log_A_cumsum = torch.cumsum(log_A, dim=1)  # B, L, D, N

    # For each position t, we need prod_{k=0}^{t} A_k
    # and for contribution from s, we need prod_{k=s+1}^{t} A_k = A_cumsum[t] / A_cumsum[s]

    # Compute A_cumsum[t] / A_cumsum[s-1] in log space
    # log(prod_{k=s}^{t} A_k) = log_A_cumsum[t] - log_A_cumsum[s-1]

    # Efficient parallel computation using the identity:
    # h_t = A_cumsum[t] * sum_{s=0}^{t} B_x[s] / A_cumsum[s]

    # Compute B_x / A_cumsum (contribution weighted by inverse cumulative A)
    A_cumsum = torch.exp(log_A_cumsum)  # B, L, D, N

    # Shift A_cumsum for proper indexing: A_cumsum_shifted[t] = prod_{k=0}^{t-1} A_k
    A_cumsum_shifted = torch.cat([
        torch.ones(B, 1, D, N, device=A.device, dtype=A.dtype),
        A_cumsum[:, :-1]
    ], dim=1)

    # Weight inputs by inverse cumulative product
    weighted_inputs = B_x / (A_cumsum_shifted + 1e-6)  # B, L, D, N

    # Cumulative sum of weighted inputs
    weighted_sum = torch.cumsum(weighted_inputs, dim=1)  # B, L, D, N

    # Final hidden states
    h = A_cumsum_shifted * weighted_sum  # B, L, D, N

    return h


class SelectiveSSMFast(nn.Module):
    """Fast Selective SSM using parallel scan.

    Key optimization: Replace sequential O(L) loop with O(log L) parallel scan.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )

        # SSM parameters - input-dependent (selective)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # Learnable SSM parameters
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Initialize A with negative values for stability
        with torch.no_grad():
            self.A.copy_(-torch.exp(torch.linspace(0, 4, d_state)).unsqueeze(0).expand(self.d_inner, -1))

    def forward(self, x):
        B, L, D = x.shape

        # Input projection: split into main and gate
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)

        # Convolution for local context
        x_conv = x_main.transpose(1, 2)  # B, d_inner, L
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)  # B, L, d_inner
        x_conv = F.silu(x_conv)

        # Input-dependent SSM parameters
        x_ssm = self.x_proj(x_conv)  # B, L, d_state*2 + 1
        dt, B_input, C = x_ssm[:, :, :1], x_ssm[:, :, 1:self.d_state+1], x_ssm[:, :, self.d_state+1:]

        # Project dt to d_inner dimensions
        dt = F.softplus(self.dt_proj(dt))  # B, L, d_inner

        # Discretize: A_bar = exp(dt * A)
        A = self.A  # d_inner, d_state
        A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # B, L, d_inner, d_state

        # B_bar * x contribution
        B_bar = dt.unsqueeze(-1) * B_input.unsqueeze(2)  # B, L, d_inner, d_state
        B_x = B_bar * x_conv.unsqueeze(-1)  # B, L, d_inner, d_state

        # FAST: Parallel scan instead of sequential loop
        h = parallel_scan(A_bar, B_x)  # B, L, d_inner, d_state

        # Output: y = C * h + D * x
        y = (h * C.unsqueeze(2)).sum(-1) + self.D.unsqueeze(0).unsqueeze(0) * x_conv  # B, L, d_inner

        # Gate and output
        y = y * F.silu(z)
        y = self.out_proj(y)
        return self.dropout(y)


class MambaBlockFast(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSMFast(d_model, d_state=d_state)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = x + self.ssm(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class Transformer_MambaFast(nn.Module):
    """Fast Mamba using parallel scan SSM.

    Same architecture as Transformer_Mamba but with O(log L) parallel scan
    instead of O(L) sequential loop.
    """
    def __init__(self, d_model, vocab_size, n_layers, n_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        # No positional embedding - SSM captures position implicitly
        self.blocks = nn.ModuleList([MambaBlockFast(d_model, d_state=16) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, N = x.shape
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))
'''


def main():
    print("=" * 70)
    print("FAST MAMBA BENCHMARK: Parallel Scan vs Sequential")
    print("=" * 70)
    print("Testing parallel scan implementation for Mamba speedup")
    print()
    sys.stdout.flush()

    # Connect to DB
    db_path = Path(__file__).parent.parent / "arcfusion.db"
    db = ArcFusionDB(str(db_path))

    # Get baseline from existing Mamba run
    mamba_runs = db.list_training_runs(model_name="Transformer_Mamba", success_only=True, limit=1)
    if mamba_runs:
        baseline_ppl = mamba_runs[0].perplexity
        baseline_time = mamba_runs[0].time_seconds
        print(f"Baseline (Sequential Mamba): {baseline_ppl:.1f} PPL in {baseline_time:.1f}s")
    else:
        baseline_ppl = 200.0  # Estimate
        baseline_time = 706.0  # From gemini1.md
        print(f"Baseline (estimated): {baseline_ppl:.1f} PPL in {baseline_time:.1f}s")

    print()
    sys.stdout.flush()

    # Run fast Mamba
    model_name = "Transformer_MambaFast"

    # Check if already run
    existing = db.list_training_runs(model_name=model_name, success_only=True, limit=1)
    if existing:
        print(f"SKIP: {model_name} already exists ({existing[0].perplexity:.1f} PPL in {existing[0].time_seconds:.1f}s)")
        speedup = baseline_time / existing[0].time_seconds
        print(f"Speedup: {speedup:.2f}x")
        return

    print(f"Training {model_name}...")
    sys.stdout.flush()

    with app.run():
        result = train_model.remote(MAMBA_FAST_CODE, model_name, CONFIG)

    if result["success"]:
        ppl = result["perplexity"]
        time_s = result["time_seconds"]
        speedup = baseline_time / time_s
        ppl_diff = ((ppl - baseline_ppl) / baseline_ppl) * 100

        print(f"\n{'=' * 70}")
        print("RESULTS")
        print("=" * 70)
        print(f"Fast Mamba: {ppl:.1f} PPL in {time_s:.1f}s")
        print(f"Sequential Mamba: {baseline_ppl:.1f} PPL in {baseline_time:.1f}s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"PPL change: {ppl_diff:+.1f}%")

        # Get a baseline run for DB comparison
        baseline_runs = db.list_training_runs(model_name="Transformer_MHA", success_only=True, limit=1)
        baseline_run_id = baseline_runs[0].run_id if baseline_runs else ""

        run_id = save_result_to_db(db, result, CONFIG, baseline_run_id, model_code=MAMBA_FAST_CODE)
        print(f"Saved: {run_id}")

        # Generate insight if significant speedup
        if speedup > 1.5:
            insight_id = generate_auto_insight(db, run_id, model_name, ppl, time_s, baseline_ppl)
            if insight_id:
                print(f"Insight: {insight_id}")
    else:
        print(f"FAILED: {result['error']}")

    print()
    sys.stdout.flush()


if __name__ == "__main__":
    main()
