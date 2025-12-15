#!/usr/bin/env python3
"""
WikiText-2 Language Model Benchmark - Compare attention mechanisms.
Run with: .venv-modal/bin/python scripts/train_wikitext.py
"""

import modal
import json
from pathlib import Path

app = modal.App("arcfusion-wikitext")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "numpy", "datasets", "tiktoken")
)

CONFIG = {
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 8,
    "batch_size": 32,
    "seq_len": 128,
    "learning_rate": 3e-4,
    "max_steps": 2000,  # Step-based training (faster than epochs)
    "eval_interval": 200,
}


@app.function(image=image, gpu="T4", timeout=1800)
def train_model(model_type: str, config: dict) -> dict:
    """Train a model on WikiText-2."""
    import time
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from datasets import load_dataset
    import tiktoken

    # Model definitions inside the function
    def build_mha(vocab_size, d_model, n_layers, n_heads):
        class MHA(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                self.n_heads = n_heads
                self.head_dim = d_model // n_heads
                self.scale = self.head_dim ** -0.5
                self.qkv = nn.Linear(d_model, 3 * d_model)
                self.out = nn.Linear(d_model, d_model)
                self.dropout = nn.Dropout(0.1)

            def forward(self, x):
                B, N, C = x.shape
                qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.masked_fill(mask, float('-inf'))
                attn = F.softmax(attn, dim=-1)
                attn = self.dropout(attn)
                return self.out((attn @ v).transpose(1, 2).reshape(B, N, C))

        class Block(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                self.ln1 = nn.LayerNorm(d_model)
                self.attn = MHA(d_model, n_heads)
                self.ln2 = nn.LayerNorm(d_model)
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_model * 4), nn.GELU(),
                    nn.Linear(d_model * 4, d_model), nn.Dropout(0.1),
                )
            def forward(self, x):
                x = x + self.attn(self.ln1(x))
                return x + self.ffn(self.ln2(x))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, d_model)
                self.pos = nn.Embedding(2048, d_model)
                self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
                self.ln_f = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, vocab_size, bias=False)
                self.head.weight = self.embed.weight
            def forward(self, x):
                B, T = x.shape
                x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
                for b in self.blocks:
                    x = b(x)
                return self.head(self.ln_f(x))
        return Model()

    def build_gqa(vocab_size, d_model, n_layers, n_heads, n_kv_heads=2):
        class GQA(nn.Module):
            def __init__(self, d_model, n_heads, n_kv_heads):
                super().__init__()
                self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
                self.head_dim = d_model // n_heads
                self.scale = self.head_dim ** -0.5
                self.q = nn.Linear(d_model, d_model)
                self.k = nn.Linear(d_model, n_kv_heads * self.head_dim)
                self.v = nn.Linear(d_model, n_kv_heads * self.head_dim)
                self.out = nn.Linear(d_model, d_model)
            def forward(self, x):
                B, N, C = x.shape
                q = self.q(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
                k = self.k(x).reshape(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
                v = self.v(x).reshape(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
                k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
                v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
                mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.masked_fill(mask, float('-inf'))
                attn = F.softmax(attn, dim=-1)
                return self.out((attn @ v).transpose(1, 2).reshape(B, N, C))

        class Block(nn.Module):
            def __init__(self, d_model, n_heads, n_kv_heads):
                super().__init__()
                self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
                self.attn = GQA(d_model, n_heads, n_kv_heads)
                self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
            def forward(self, x):
                return x + self.ffn(self.ln2(x + self.attn(self.ln1(x))))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, d_model)
                self.pos = nn.Embedding(2048, d_model)
                self.blocks = nn.ModuleList([Block(d_model, n_heads, n_kv_heads) for _ in range(n_layers)])
                self.ln_f = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, vocab_size, bias=False)
                self.head.weight = self.embed.weight
            def forward(self, x):
                x = self.embed(x) + self.pos(torch.arange(x.shape[1], device=x.device))
                for b in self.blocks: x = b(x)
                return self.head(self.ln_f(x))
        return Model()

    def build_mqa(vocab_size, d_model, n_layers, n_heads):
        class MQA(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                self.n_heads = n_heads
                self.head_dim = d_model // n_heads
                self.scale = self.head_dim ** -0.5
                self.q = nn.Linear(d_model, d_model)
                self.k = nn.Linear(d_model, self.head_dim)
                self.v = nn.Linear(d_model, self.head_dim)
                self.out = nn.Linear(d_model, d_model)
            def forward(self, x):
                B, N, C = x.shape
                q = self.q(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
                k = self.k(x).reshape(B, N, 1, self.head_dim).transpose(1, 2).expand(-1, self.n_heads, -1, -1)
                v = self.v(x).reshape(B, N, 1, self.head_dim).transpose(1, 2).expand(-1, self.n_heads, -1, -1)
                mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.masked_fill(mask, float('-inf'))
                attn = F.softmax(attn, dim=-1)
                return self.out((attn @ v).transpose(1, 2).reshape(B, N, C))

        class Block(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
                self.attn = MQA(d_model, n_heads)
                self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
            def forward(self, x):
                return x + self.ffn(self.ln2(x + self.attn(self.ln1(x))))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, d_model)
                self.pos = nn.Embedding(2048, d_model)
                self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
                self.ln_f = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, vocab_size, bias=False)
                self.head.weight = self.embed.weight
            def forward(self, x):
                x = self.embed(x) + self.pos(torch.arange(x.shape[1], device=x.device))
                for b in self.blocks: x = b(x)
                return self.head(self.ln_f(x))
        return Model()

    result = {
        "success": False,
        "model_type": model_type,
        "test_loss": float("inf"),
        "test_ppl": float("inf"),
        "parameters": 0,
        "time_seconds": 0.0,
        "error": None,
    }

    start = time.time()

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        print(f"Training: {model_type}")

        # Load WikiText-2
        print("Loading WikiText-2...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        enc = tiktoken.get_encoding("gpt2")
        vocab_size = enc.n_vocab

        def tokenize_split(split):
            text = "\n".join([x["text"] for x in ds[split] if x["text"].strip()])
            return enc.encode(text)

        train_tokens = tokenize_split("train")
        val_tokens = tokenize_split("validation")
        test_tokens = tokenize_split("test")
        print(f"Tokens: train={len(train_tokens):,}, val={len(val_tokens):,}, test={len(test_tokens):,}")

        # Dataset
        class TokenDataset(Dataset):
            def __init__(self, tokens, seq_len):
                self.tokens = torch.tensor(tokens, dtype=torch.long)
                self.seq_len = seq_len
            def __len__(self):
                return max(0, len(self.tokens) - self.seq_len - 1)
            def __getitem__(self, idx):
                c = self.tokens[idx:idx + self.seq_len + 1]
                return c[:-1], c[1:]

        sl, bs = config["seq_len"], config["batch_size"]
        train_loader = DataLoader(TokenDataset(train_tokens, sl), batch_size=bs, shuffle=True, drop_last=True)
        val_loader = DataLoader(TokenDataset(val_tokens, sl), batch_size=bs, drop_last=True)
        test_loader = DataLoader(TokenDataset(test_tokens, sl), batch_size=bs, drop_last=True)

        # Build model
        print(f"Building {model_type}...")
        d, nl, nh = config["d_model"], config["n_layers"], config["n_heads"]
        if model_type == "MHA":
            model = build_mha(vocab_size, d, nl, nh)
        elif model_type == "GQA":
            model = build_gqa(vocab_size, d, nl, nh, n_kv_heads=2)
        else:
            model = build_mqa(vocab_size, d, nl, nh)

        model = model.to(device)
        result["parameters"] = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {result['parameters']:,}")

        # Training with mixed precision
        opt = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        crit = nn.CrossEntropyLoss()
        max_steps = config["max_steps"]
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_steps)
        scaler = torch.amp.GradScaler('cuda')

        def evaluate(loader, max_batches=50):
            model.eval()
            tl, tt = 0, 0
            with torch.no_grad():
                for i, (x, y) in enumerate(loader):
                    if i >= max_batches:
                        break
                    x, y = x.to(device), y.to(device)
                    with torch.amp.autocast('cuda'):
                        loss = crit(model(x).reshape(-1, vocab_size), y.reshape(-1))
                    tl += loss.item() * y.numel()
                    tt += y.numel()
            return tl / tt if tt > 0 else float('inf')

        print(f"Training {max_steps} steps with mixed precision...")
        step = 0
        train_iter = iter(train_loader)
        model.train()
        while step < max_steps:
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = crit(model(x).reshape(-1, vocab_size), y.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1

            if step % config["eval_interval"] == 0:
                vl = evaluate(val_loader)
                print(f"Step {step}: train_loss={loss.item():.4f}, val_ppl={math.exp(vl):.2f}")
                model.train()

        print(f"Completed {max_steps} steps")

        result["test_loss"] = evaluate(test_loader)
        result["test_ppl"] = math.exp(result["test_loss"])
        result["success"] = True
        print(f"Test PPL: {result['test_ppl']:.2f}")

    except Exception as e:
        import traceback
        result["error"] = str(e) + "\n" + traceback.format_exc()
        print(f"ERROR: {e}")
        print(traceback.format_exc())

    result["time_seconds"] = time.time() - start
    return result


def main():
    print("=" * 60)
    print("WikiText-2 LM Benchmark: MHA vs GQA vs MQA")
    print("=" * 60)

    results = {}
    for m in ["MHA", "GQA", "MQA"]:
        print(f"\n--- {m} ---")
        with app.run():
            results[m] = train_model.remote(m, CONFIG)
        if results[m]["success"]:
            print(f"{m}: {results[m]['test_ppl']:.2f} ppl")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for m in sorted(results, key=lambda k: results[k].get("test_ppl", 9999)):
        r = results[m]
        if r["success"]:
            print(f"{m}: {r['parameters']:,} params, {r['test_ppl']:.2f} ppl, {r['time_seconds']:.0f}s")

    Path("experiments").mkdir(exist_ok=True)
    with open("experiments/wikitext_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
