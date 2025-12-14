#!/usr/bin/env python3
"""
FAIR cloud training comparison - all models have same structure.
Only the attention mechanism differs.

Run with: .venv-modal/bin/python scripts/cloud_train_fair.py

Features:
- Mixed precision (FP16) for ~2x speedup on tensor cores
- A10G GPU (faster than T4, good FP16 performance)
- Always trains vanilla Transformer_MHA as baseline first
"""

import modal
import json
from pathlib import Path

app = modal.App("arcfusion-fair-compare")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "numpy", "datasets", "tiktoken")
)

# Training configuration
CONFIG = {
    "d_model": 256,
    "vocab_size": 50257,  # GPT-2 tokenizer vocab size
    "n_layers": 4,
    "n_heads": 8,
    "seq_len": 128,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "max_steps": 2000,  # Fewer steps but on real data
    "eval_interval": 200,
    "mixed_precision": True,  # Use FP16 autocast + GradScaler
    "gpu": "A10G",  # A10G has better FP16 perf than T4
    "dataset": "wikitext-2",  # Real text data
}

# Baseline model - always trained first for comparison
BASELINE_MODEL = "Transformer_MHA"


@app.function(image=image, gpu="A10G", timeout=600)
def train_model(code: str, model_name: str, config: dict) -> dict:
    """Train a model on WikiText-2 with mixed precision."""
    import time
    import math
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from datasets import load_dataset
    import tiktoken

    use_amp = config.get("mixed_precision", True)

    result = {
        "success": False,
        "model_name": model_name,
        "final_train_loss": float("inf"),
        "eval_loss": float("inf"),
        "perplexity": float("inf"),
        "steps": 0,
        "time_seconds": 0.0,
        "parameters": 0,
        "error": None,
        "gpu": config.get("gpu", "A10G"),
        "mixed_precision": use_amp,
        "dataset": config.get("dataset", "wikitext-2"),
    }

    start = time.time()

    try:
        # Load and tokenize WikiText-2
        print("Loading WikiText-2 dataset...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        enc = tiktoken.get_encoding("gpt2")

        # Tokenize all splits
        def tokenize_split(split_name):
            texts = [t for t in ds[split_name]['text'] if t.strip()]
            all_tokens = []
            for text in texts:
                all_tokens.extend(enc.encode(text))
            return all_tokens

        train_tokens = tokenize_split('train')
        val_tokens = tokenize_split('validation')
        test_tokens = tokenize_split('test')
        print(f"Tokens: train={len(train_tokens):,}, val={len(val_tokens):,}, test={len(test_tokens):,}")

        # Create dataset class
        class TokenDataset(Dataset):
            def __init__(self, tokens, seq_len):
                self.tokens = torch.tensor(tokens, dtype=torch.long)
                self.seq_len = seq_len

            def __len__(self):
                return max(0, len(self.tokens) - self.seq_len - 1)

            def __getitem__(self, idx):
                chunk = self.tokens[idx:idx + self.seq_len + 1]
                return chunk[:-1], chunk[1:]

        seq_len = config["seq_len"]
        batch_size = config["batch_size"]
        train_ds = TokenDataset(train_tokens, seq_len)
        val_ds = TokenDataset(val_tokens, seq_len)
        test_ds = TokenDataset(test_tokens, seq_len)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)

        # Execute model code
        ns = {}
        exec(code, ns)
        model_class = ns.get(model_name)
        if not model_class:
            result["error"] = f"Class {model_name} not found"
            return result

        device = "cuda" if torch.cuda.is_available() else "cpu"
        amp_status = "with mixed precision (FP16)" if use_amp else "with FP32"
        print(f"Training {model_name} on {device} {amp_status}")

        model = model_class(
            d_model=config["d_model"],
            vocab_size=config["vocab_size"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
        ).to(device)

        result["parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {result['parameters']:,}")

        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        vocab_size = config["vocab_size"]

        def evaluate(loader, max_batches=50):
            model.eval()
            total_loss, total_tokens = 0, 0
            with torch.no_grad():
                for i, (x, y) in enumerate(loader):
                    if i >= max_batches:
                        break
                    x, y = x.to(device), y.to(device)
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        logits = model(x)
                        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                    total_loss += loss.item() * y.numel()
                    total_tokens += y.numel()
            return total_loss / total_tokens if total_tokens > 0 else float('inf')

        # Training loop
        print(f"Training for {config['max_steps']} steps...")
        step = 0
        train_iter = iter(train_loader)
        model.train()

        while step < config["max_steps"]:
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(x)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1

            if step % config["eval_interval"] == 0:
                val_loss = evaluate(val_loader)
                val_ppl = math.exp(min(val_loss, 20))
                print(f"Step {step}: train_loss={loss.item():.4f}, val_ppl={val_ppl:.2f}")
                model.train()

            result["steps"] = step
            result["final_train_loss"] = loss.item()

        # Final evaluation on test set
        result["eval_loss"] = evaluate(test_loader, max_batches=100)
        result["perplexity"] = math.exp(min(result["eval_loss"], 20))
        result["success"] = True
        print(f"Final test perplexity: {result['perplexity']:.2f}")

    except Exception as e:
        import traceback
        result["error"] = f"{e}\n{traceback.format_exc()}"

    result["time_seconds"] = time.time() - start
    return result


# All models have IDENTICAL structure: Embed → N x (Attn + FFN with residuals) → Output
# Only the attention mechanism differs

MODELS = {
    # Standard Multi-Head Attention
    "Transformer_MHA": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):
    """Standard Multi-Head Attention with causal masking"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Causal mask: prevent attending to future tokens
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(x)

class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer_MHA(nn.Module):
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
''',

    # Grouped Query Attention (like LLaMA 2)
    "Transformer_GQA": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    """Grouped Query Attention with causal masking - fewer KV heads than Q heads"""
    def __init__(self, d_model, n_heads, n_kv_heads=2, dropout=0.1):
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
        # Causal mask: prevent attending to future tokens
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        return self.out(x)

class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GQA(d_model, n_heads, n_kv_heads=2)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer_GQA(nn.Module):
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
''',

    # Multi-Query Attention (like Falcon)
    "Transformer_MQA": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MQA(nn.Module):
    """Multi-Query Attention with causal masking - single KV head shared by all Q heads"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.head_dim)  # Single head
        self.v_proj = nn.Linear(d_model, self.head_dim)  # Single head
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(x).reshape(B, N, 1, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(x).reshape(B, N, 1, self.head_dim).transpose(1, 2).contiguous()

        # Repeat K,V for all heads (use repeat for proper gradient flow)
        k = k.repeat(1, self.n_heads, 1, 1)
        v = v.repeat(1, self.n_heads, 1, 1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Causal mask: prevent attending to future tokens
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        return self.out(x)

class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MQA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer_MQA(nn.Module):
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
''',
}


def main():
    print("=" * 70)
    print("FAIR COMPARISON - Same structure, different attention mechanisms")
    print("=" * 70)
    print(f"Config: {CONFIG['n_layers']} layers, d_model={CONFIG['d_model']}, {CONFIG['max_steps']} steps")
    print(f"GPU: {CONFIG['gpu']}, Mixed Precision: {CONFIG['mixed_precision']}")
    print()

    results = {}

    # Always train baseline first
    print(f"\n{'='*70}")
    print(f"Training BASELINE: {BASELINE_MODEL}")
    print("=" * 70)

    baseline_code = MODELS[BASELINE_MODEL]
    with app.run():
        baseline_result = train_model.remote(baseline_code, BASELINE_MODEL, CONFIG)

    results[BASELINE_MODEL] = baseline_result
    baseline_result["is_baseline"] = True

    if baseline_result["success"]:
        print(f"\nBASELINE Results ({BASELINE_MODEL}):")
        print(f"  Parameters: {baseline_result['parameters']:,}")
        print(f"  Train Loss: {baseline_result['final_train_loss']:.4f}")
        print(f"  Eval Loss:  {baseline_result['eval_loss']:.4f}")
        print(f"  Perplexity: {baseline_result['perplexity']:.2f}")
        print(f"  Time: {baseline_result['time_seconds']:.1f}s")
        baseline_ppl = baseline_result["perplexity"]
    else:
        print(f"\nBASELINE FAILED: {baseline_result['error']}")
        baseline_ppl = None

    # Train other models
    for name, code in MODELS.items():
        if name == BASELINE_MODEL:
            continue  # Already trained

        print(f"\n{'='*70}")
        print(f"Training: {name}")
        print("=" * 70)

        with app.run():
            result = train_model.remote(code, name, CONFIG)

        result["is_baseline"] = False
        results[name] = result

        if result["success"]:
            print(f"\nResults for {name}:")
            print(f"  Parameters: {result['parameters']:,}")
            print(f"  Train Loss: {result['final_train_loss']:.4f}")
            print(f"  Eval Loss:  {result['eval_loss']:.4f}")
            print(f"  Perplexity: {result['perplexity']:.2f}")
            print(f"  Time: {result['time_seconds']:.1f}s")

            # Compare to baseline
            if baseline_ppl is not None:
                delta = result["perplexity"] - baseline_ppl
                pct = (delta / baseline_ppl) * 100
                if delta < 0:
                    print(f"  vs Baseline: {delta:.2f} ({pct:+.1f}%) BETTER")
                else:
                    print(f"  vs Baseline: +{delta:.2f} ({pct:+.1f}%) worse")
        else:
            print(f"\nFAILED: {result['error']}")

    # Summary with baseline comparison
    print("\n" + "=" * 70)
    print("FAIR COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Model':<20} {'Params':>12} {'Eval Loss':>12} {'Perplexity':>12} {'vs Baseline':>14}")
    print("-" * 70)

    sorted_results = sorted(
        [(k, v) for k, v in results.items() if v["success"]],
        key=lambda x: x[1]["eval_loss"]
    )

    for name, r in sorted_results:
        if baseline_ppl and r["success"]:
            delta = r["perplexity"] - baseline_ppl
            pct = (delta / baseline_ppl) * 100
            vs_baseline = f"{pct:+.1f}%" if name != BASELINE_MODEL else "(baseline)"
        else:
            vs_baseline = "N/A"
        print(f"{name:<20} {r['parameters']:>12,} {r['eval_loss']:>12.4f} {r['perplexity']:>12.2f} {vs_baseline:>14}")

    # Save with metadata
    output = {
        "config": CONFIG,
        "baseline_model": BASELINE_MODEL,
        "results": results,
    }
    Path("experiments").mkdir(exist_ok=True)
    with open("experiments/fair_comparison_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to experiments/fair_comparison_results.json")


if __name__ == "__main__":
    main()
