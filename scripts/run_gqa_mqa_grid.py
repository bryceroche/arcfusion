#!/usr/bin/env python3
"""
GQA/MQA Variant Grid: Systematic sweep over kv_heads and layers

Grid:
- kv_heads: 1 (MQA), 2 (GQA), 4 (GQA4)
- layers: 10, 12, 14, 18

Run with: PYTHONUNBUFFERED=1 .venv-modal/bin/python scripts/run_gqa_mqa_grid.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "arcfusion"))

from cloud_train_fair import CONFIG, app, train_model, save_result_to_db, generate_auto_insight
from db import ArcFusionDB


def generate_model_code(n_layers: int, n_kv_heads: int) -> tuple[str, str]:
    """Generate model code with variable kv_heads and layers.

    Returns: (code, model_class_name)
    """
    # Name based on kv_heads: 1=MQA, 2=GQA, 4=GQA4, 8=MHA
    if n_kv_heads == 1:
        attn_name = "MQA"
    elif n_kv_heads == 2:
        attn_name = "GQA"
    elif n_kv_heads == 4:
        attn_name = "GQA4"
    else:
        attn_name = f"KV{n_kv_heads}"

    model_name = f"Transformer_{attn_name}{n_layers}"

    code = f'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with variable kv_heads"""
    def __init__(self, d_model, n_heads, n_kv_heads={n_kv_heads}, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(x).reshape(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(x).reshape(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # Repeat KV heads to match Q heads
        n_rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(n_rep, dim=1).contiguous()
        v = v.repeat_interleave(n_rep, dim=1).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads={n_kv_heads})
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class {model_name}(nn.Module):
    """Transformer with {attn_name} attention ({n_kv_heads} KV heads) and {n_layers} layers"""
    def __init__(self, d_model, vocab_size, n_layers, n_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(2048, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range({n_layers})])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, N = x.shape
        x = self.embed(x) + self.pos(torch.arange(N, device=x.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))
'''
    return code, model_name


# Grid configuration
GRID = [
    # (n_layers, n_kv_heads)
    # MQA variants (kv_heads=1)
    (10, 1), (12, 1), (14, 1), (18, 1),
    # GQA variants (kv_heads=2) - some already run, but include for completeness
    (10, 2), (12, 2), (14, 2), (18, 2),
    # GQA4 variants (kv_heads=4)
    (10, 4), (12, 4), (14, 4), (18, 4),
]


def main():
    print("=" * 70)
    print("GQA/MQA VARIANT GRID EXPERIMENT")
    print("=" * 70)
    print("Grid: kv_heads ∈ {1, 2, 4} × layers ∈ {10, 12, 14, 18}")
    print(f"Total experiments: {len(GRID)}")
    print()
    sys.stdout.flush()

    # Connect to DB
    db_path = Path(__file__).parent.parent / "arcfusion.db"
    db = ArcFusionDB(str(db_path))

    # Get baseline for comparison
    baseline_runs = db.list_training_runs(model_name="Transformer_MHA", success_only=True, limit=1)
    baseline_run_id = baseline_runs[0].run_id if baseline_runs else ""
    baseline_ppl = baseline_runs[0].perplexity if baseline_runs else 274.8

    print(f"Baseline: MHA (4-layer) = {baseline_ppl:.1f} PPL")
    print()
    sys.stdout.flush()

    results = []

    for n_layers, n_kv_heads in GRID:
        code, model_name = generate_model_code(n_layers, n_kv_heads)

        # Check if already run
        existing = db.list_training_runs(model_name=model_name, success_only=True, limit=1)
        if existing:
            print(f"SKIP: {model_name} already exists ({existing[0].perplexity:.1f} PPL)")
            results.append({
                "name": model_name,
                "layers": n_layers,
                "kv_heads": n_kv_heads,
                "ppl": existing[0].perplexity,
                "time": 0,  # Unknown from existing
                "skipped": True,
            })
            continue

        print(f"\n{'=' * 70}")
        print(f"Training {model_name} ({n_layers}L, {n_kv_heads} KV heads)...")
        print("=" * 70)
        sys.stdout.flush()

        with app.run():
            result = train_model.remote(code, model_name, CONFIG)

        if result["success"]:
            ppl = result["perplexity"]
            time_s = result["time_seconds"]
            vs_baseline = ((ppl - baseline_ppl) / baseline_ppl) * 100

            print(f"  PPL: {ppl:.1f}")
            print(f"  Time: {time_s:.1f}s")
            print(f"  vs baseline: {vs_baseline:+.1f}%")

            run_id = save_result_to_db(db, result, CONFIG, baseline_run_id, model_code=code)
            print(f"  Saved: {run_id}")

            # Auto-generate insight if notable
            insight_id = generate_auto_insight(db, run_id, model_name, ppl, time_s, baseline_ppl)
            if insight_id:
                print(f"  Insight: {insight_id}")

            results.append({
                "name": model_name,
                "layers": n_layers,
                "kv_heads": n_kv_heads,
                "ppl": ppl,
                "time": time_s,
                "skipped": False,
            })
        else:
            print(f"  FAILED: {result['error']}")
            results.append({
                "name": model_name,
                "layers": n_layers,
                "kv_heads": n_kv_heads,
                "ppl": float('inf'),
                "time": 0,
                "skipped": False,
                "error": result['error'],
            })

        sys.stdout.flush()

    # Final summary
    print("\n" + "=" * 70)
    print("GQA/MQA GRID RESULTS")
    print("=" * 70)

    # Group by kv_heads for comparison
    print(f"\n{'Layers':>6} | {'MQA (kv=1)':>12} | {'GQA (kv=2)':>12} | {'GQA4 (kv=4)':>12}")
    print("-" * 55)

    for layers in [10, 12, 14, 18]:
        row = f"{layers:>6} |"
        for kv in [1, 2, 4]:
            r = next((x for x in results if x["layers"] == layers and x["kv_heads"] == kv), None)
            if r and r["ppl"] < float('inf'):
                row += f" {r['ppl']:>10.1f} |"
            else:
                row += f" {'---':>10} |"
        print(row)

    # Pareto frontier analysis
    print("\n" + "=" * 70)
    print("PARETO FRONTIER (Best PPL per time bucket)")
    print("=" * 70)

    # Sort by time and find frontier
    valid_results = [r for r in results if r.get("time", 0) > 0 and r["ppl"] < float('inf')]
    if valid_results:
        valid_results.sort(key=lambda x: x["time"])

        frontier = []
        best_ppl = float('inf')
        for r in valid_results:
            if r["ppl"] < best_ppl:
                frontier.append(r)
                best_ppl = r["ppl"]

        print(f"\n{'Model':<25} {'PPL':>8} {'Time':>8} {'KV Heads':>10}")
        print("-" * 55)
        for r in frontier:
            print(f"{r['name']:<25} {r['ppl']:>8.1f} {r['time']:>7.0f}s {r['kv_heads']:>10}")

    print()
    sys.stdout.flush()


if __name__ == "__main__":
    main()
