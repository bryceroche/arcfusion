#!/usr/bin/env python3
"""
FAIR cloud training comparison - all models have same structure.
Only the attention mechanism differs.

Run with: .venv-modal/bin/python scripts/cloud_train_fair.py

Features:
- Mixed precision (FP16) for ~2x speedup on tensor cores
- A10G GPU (faster than T4, good FP16 performance)
- Baseline caching: trains vanilla Transformer once, reuses result
- Results saved to arcfusion.db training_runs table
"""
from __future__ import annotations

import modal
import json
import sys
from pathlib import Path
from datetime import datetime

# DB imports are deferred to main() to avoid Modal serialization issues

app = modal.App("arcfusion-fair-compare")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "numpy", "datasets", "tiktoken")
    # Pre-cache WikiText-2 during image build to avoid runtime download timeouts
    .run_commands(
        'python -c "from datasets import load_dataset; load_dataset(\'wikitext\', \'wikitext-2-raw-v1\', trust_remote_code=True)"'
    )
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
    "gpu": "A100",  # Default GPU for experiments
    "dataset": "wikitext-2",  # Real text data
    "seed": 42,  # Default seed (overridden for baselines)
}

# Baseline configuration
BASELINE_MODEL = "Transformer_MHA"
BASELINE_GPU = "A100"  # Use A100 for speed
BASELINE_TARGET_RUNS = 3  # How many baseline runs to average
BASELINE_SEEDS = [42, 123, 456]  # Seeds for baseline runs


@app.function(image=image, gpu="A100", timeout=1800)
def train_model(code: str, model_name: str, config: dict) -> dict:
    """Train model on A100 GPU (fast experiments)."""
    import time
    import math
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from datasets import load_dataset
    import tiktoken

    gpu_type = "A100"
    use_amp = config.get("mixed_precision", True)
    seed = config.get("seed", 42)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
        "gpu": gpu_type,
        "mixed_precision": use_amp,
        "dataset": config.get("dataset", "wikitext-2"),
        "seed": seed,
    }

    start = time.time()

    try:
        print(f"Loading WikiText-2 dataset (seed={seed})...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        enc = tiktoken.get_encoding("gpt2")

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

    # Mamba-style Selective State Space Model
    "Transformer_Mamba": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    """Simplified Selective State Space Model (Mamba-style).

    Key ideas from Mamba paper:
    - Input-dependent (selective) state transitions
    - Linear time complexity O(n) vs O(n^2) for attention
    - Parallel scan for efficient training
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
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # dt, B, C

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
        x_conv = self.conv1d(x_conv)[:, :, :L]  # Causal: trim to original length
        x_conv = x_conv.transpose(1, 2)  # B, L, d_inner
        x_conv = F.silu(x_conv)

        # Input-dependent SSM parameters
        x_ssm = self.x_proj(x_conv)  # B, L, d_state*2 + 1
        dt, B_input, C = x_ssm[:, :, :1], x_ssm[:, :, 1:self.d_state+1], x_ssm[:, :, self.d_state+1:]

        # Project dt to d_inner dimensions
        dt = F.softplus(self.dt_proj(dt))  # B, L, d_inner

        # Discretize: A_bar = exp(dt * A)
        A = self.A  # d_inner, d_state

        # Simplified parallel SSM computation (not full parallel scan, but efficient)
        # This is a simplified version - full Mamba uses custom CUDA kernels
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(L):
            dt_t = dt[:, t, :]  # B, d_inner
            B_t = B_input[:, t, :]  # B, d_state
            C_t = C[:, t, :]  # B, d_state
            x_t = x_conv[:, t, :]  # B, d_inner

            # State update: h = A_bar * h + B_bar * x
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # B, d_inner, d_state
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # B, d_inner, d_state
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)

            # Output: y = C * h + D * x
            y_t = (h * C_t.unsqueeze(1)).sum(-1) + self.D * x_t  # B, d_inner
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # B, L, d_inner

        # Gate and output
        y = y * F.silu(z)
        y = self.out_proj(y)
        return self.dropout(y)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.ssm(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class Transformer_Mamba(nn.Module):
    """Mamba-style language model using Selective SSM instead of attention.

    Note: n_heads is ignored (kept for API compatibility).
    Uses d_state = d_model // 16 for state dimension.
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        d_state = max(16, d_model // 16)  # Scale state dim with model
        self.embed = nn.Embedding(vocab_size, d_model)
        # No positional embedding - SSM captures position implicitly
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x)  # No positional encoding needed
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
''',

    # Hybrid Transformer-Mamba (Jamba-style)
    "Transformer_Hybrid": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ MHA Block (from Transformer_MHA) ============
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
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(x)

class MHABlock(nn.Module):
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
        x = x + self.ffn(self.ln2(x))
        return x

# ============ SSM Block (from Transformer_Mamba) ============
class SelectiveSSM(nn.Module):
    """Simplified Selective State Space Model (Mamba-style)."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            self.A.copy_(-torch.exp(torch.linspace(0, 4, d_state)).unsqueeze(0).expand(self.d_inner, -1))

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)
        x_conv = x_main.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        x_ssm = self.x_proj(x_conv)
        dt, B_input, C = x_ssm[:, :, :1], x_ssm[:, :, 1:self.d_state+1], x_ssm[:, :, self.d_state+1:]
        dt = F.softplus(self.dt_proj(dt))
        A = self.A
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            dt_t, B_t, C_t, x_t = dt[:, t, :], B_input[:, t, :], C[:, t, :], x_conv[:, t, :]
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            y_t = (h * C_t.unsqueeze(1)).sum(-1) + self.D * x_t
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y))

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model), nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.ssm(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# ============ Hybrid Model ============
class Transformer_Hybrid(nn.Module):
    """Hybrid Transformer-Mamba: alternates SSM and Attention layers.

    Architecture: Mamba -> Mamba -> MHA -> Mamba (for 4 layers)
    - SSM layers: efficient O(n) sequential processing
    - Attention layers: global reasoning every 3rd layer

    Inspired by Jamba (AI21) which showed hybrid architectures
    can outperform pure Transformer or pure Mamba.
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        d_state = max(16, d_model // 16)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)  # Keep pos embed for attention layers

        # Build hybrid stack: attention every 3rd layer
        # For 4 layers: Mamba, Mamba, MHA, Mamba
        blocks = []
        for i in range(n_layers):
            if (i + 1) % 3 == 0:  # Layer 3, 6, 9... get attention
                blocks.append(MHABlock(d_model, n_heads))
            else:
                blocks.append(MambaBlock(d_model, d_state))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
''',

    # Mamba-Heavy Hybrid (3 SSM : 1 MHA ratio)
    "Transformer_MambaHeavy": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ MHA Block ============
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
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(x)

class MHABlock(nn.Module):
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
        x = x + self.ffn(self.ln2(x))
        return x

# ============ SSM Block ============
class SelectiveSSM(nn.Module):
    """Simplified Selective State Space Model (Mamba-style)."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            self.A.copy_(-torch.exp(torch.linspace(0, 4, d_state)).unsqueeze(0).expand(self.d_inner, -1))

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)
        x_conv = x_main.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        x_ssm = self.x_proj(x_conv)
        dt, B_input, C = x_ssm[:, :, :1], x_ssm[:, :, 1:self.d_state+1], x_ssm[:, :, self.d_state+1:]
        dt = F.softplus(self.dt_proj(dt))
        A = self.A
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            dt_t, B_t, C_t, x_t = dt[:, t, :], B_input[:, t, :], C[:, t, :], x_conv[:, t, :]
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            y_t = (h * C_t.unsqueeze(1)).sum(-1) + self.D * x_t
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y))

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model), nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.ssm(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# ============ Mamba-Heavy Hybrid Model ============
class Transformer_MambaHeavy(nn.Module):
    """Mamba-Heavy Hybrid: 3 SSM layers for every 1 MHA layer.

    Architecture for 4 layers: [Mamba, Mamba, Mamba, MHA]
    - Only 1 attention layer (at the end) for final global reasoning
    - 3 Mamba layers for efficient sequential processing

    Hypothesis: Pure Mamba has best quality, but one attention layer
    at the end might help with final token predictions without
    hurting speed as much as the 2:1 ratio Hybrid.
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        d_state = max(16, d_model // 16)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)  # Keep pos embed for attention layer

        # Build 3:1 stack: MHA only on last layer
        # For 4 layers: [Mamba, Mamba, Mamba, MHA]
        blocks = []
        for i in range(n_layers):
            if i == n_layers - 1:  # Last layer gets attention
                blocks.append(MHABlock(d_model, n_heads))
            else:
                blocks.append(MambaBlock(d_model, d_state))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
''',

    # Attention-FIRST Hybrid (global context early, SSM refinement)
    "Transformer_AttnFirst": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ MHA Block ============
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
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(x)

class MHABlock(nn.Module):
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
        x = x + self.ffn(self.ln2(x))
        return x

# ============ SSM Block ============
class SelectiveSSM(nn.Module):
    """Simplified Selective State Space Model (Mamba-style)."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            self.A.copy_(-torch.exp(torch.linspace(0, 4, d_state)).unsqueeze(0).expand(self.d_inner, -1))

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)
        x_conv = x_main.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        x_ssm = self.x_proj(x_conv)
        dt, B_input, C = x_ssm[:, :, :1], x_ssm[:, :, 1:self.d_state+1], x_ssm[:, :, self.d_state+1:]
        dt = F.softplus(self.dt_proj(dt))
        A = self.A
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            dt_t, B_t, C_t, x_t = dt[:, t, :], B_input[:, t, :], C[:, t, :], x_conv[:, t, :]
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            y_t = (h * C_t.unsqueeze(1)).sum(-1) + self.D * x_t
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y))

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model), nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.ssm(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# ============ Attention-FIRST Hybrid Model ============
class Transformer_AttnFirst(nn.Module):
    """Attention-FIRST Hybrid: Global context early, SSM refinement.

    Architecture for 4 layers: [MHA, Mamba, Mamba, Mamba]
    - First layer is attention for global context capture
    - Remaining layers are Mamba for efficient sequential refinement

    Hypothesis: Providing global context FIRST allows Mamba layers
    to refine with full sequence awareness, potentially better than
    attention-at-end where Mamba has no global context to work with.
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        d_state = max(16, d_model // 16)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)  # Position embedding for attention

        # Build attention-first stack: MHA first, then Mamba
        # For 4 layers: [MHA, Mamba, Mamba, Mamba]
        blocks = []
        for i in range(n_layers):
            if i == 0:  # First layer gets attention
                blocks.append(MHABlock(d_model, n_heads))
            else:
                blocks.append(MambaBlock(d_model, d_state))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
''',

    # Mamba 4:1 Hybrid (4 SSM : 1 MHA ratio, 5 layers total)
    "Transformer_Mamba4to1": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ MHA Block ============
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
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(x)

class MHABlock(nn.Module):
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
        x = x + self.ffn(self.ln2(x))
        return x

# ============ SSM Block ============
class SelectiveSSM(nn.Module):
    """Simplified Selective State Space Model (Mamba-style)."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            self.A.copy_(-torch.exp(torch.linspace(0, 4, d_state)).unsqueeze(0).expand(self.d_inner, -1))

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)
        x_conv = x_main.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        x_ssm = self.x_proj(x_conv)
        dt, B_input, C = x_ssm[:, :, :1], x_ssm[:, :, 1:self.d_state+1], x_ssm[:, :, self.d_state+1:]
        dt = F.softplus(self.dt_proj(dt))
        A = self.A
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            dt_t, B_t, C_t, x_t = dt[:, t, :], B_input[:, t, :], C[:, t, :], x_conv[:, t, :]
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            y_t = (h * C_t.unsqueeze(1)).sum(-1) + self.D * x_t
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y))

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model), nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.ssm(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# ============ Mamba 4:1 Hybrid Model ============
class Transformer_Mamba4to1(nn.Module):
    """Mamba 4:1 Hybrid: 4 SSM layers + 1 MHA layer = 5 layers total.

    Architecture: [Mamba, Mamba, Mamba, Mamba, MHA]
    - 4 Mamba layers for efficient O(n) sequential processing
    - 1 attention layer at the end for final global reasoning

    Testing hypothesis: Can we push the Mamba ratio even higher?
    Compared to MambaHeavy (3:1), this has one more Mamba layer.
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        d_state = max(16, d_model // 16)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)

        # Build 4:1 stack: 4 Mamba + 1 MHA (ignores n_layers, always 5)
        blocks = []
        for i in range(4):  # 4 Mamba blocks
            blocks.append(MambaBlock(d_model, d_state))
        blocks.append(MHABlock(d_model, n_heads))  # 1 MHA at end
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
''',

    # Linear Attention Hybrid (O(n) attention instead of O(n²))
    "Transformer_LinearAttn": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ Linear Attention ============
class LinearAttention(nn.Module):
    """Linear Attention: O(n) complexity via kernel trick.

    Instead of softmax(QK^T)V which is O(n²), we use:
    (φ(Q) @ φ(K)^T) @ V = φ(Q) @ (φ(K)^T @ V)

    where φ is a feature map (here: elu(x) + 1).
    This allows computing in O(n*d²) which is O(n) for fixed d.

    Based on "Transformers are RNNs" (Katharopoulos et al., 2020)
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def feature_map(self, x):
        """ELU-based feature map: elu(x) + 1 to ensure positivity."""
        return F.elu(x) + 1

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, H, N, D

        # Apply feature map
        q = self.feature_map(q)
        k = self.feature_map(k)

        # Causal linear attention via cumulative sum
        # For each position i, we can only attend to positions <= i
        # We compute this efficiently using cumulative KV states

        # Compute KV: (B, H, D, D) cumulative state
        kv = torch.zeros(B, self.n_heads, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
        k_sum = torch.zeros(B, self.n_heads, self.head_dim, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(N):
            k_t = k[:, :, t, :]  # B, H, D
            v_t = v[:, :, t, :]  # B, H, D
            q_t = q[:, :, t, :]  # B, H, D

            # Update cumulative state
            kv = kv + torch.einsum('bhd,bhe->bhde', k_t, v_t)
            k_sum = k_sum + k_t

            # Compute attention output: q @ kv / (q @ k_sum)
            num = torch.einsum('bhd,bhde->bhe', q_t, kv)
            denom = torch.einsum('bhd,bhd->bh', q_t, k_sum).unsqueeze(-1) + 1e-6
            out_t = num / denom
            outputs.append(out_t)

        out = torch.stack(outputs, dim=2)  # B, H, N, D
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.dropout(self.out(out))


class LinearAttnBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = LinearAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model), nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# ============ SSM Block ============
class SelectiveSSM(nn.Module):
    """Simplified Selective State Space Model (Mamba-style)."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            self.A.copy_(-torch.exp(torch.linspace(0, 4, d_state)).unsqueeze(0).expand(self.d_inner, -1))

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)
        x_conv = x_main.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        x_ssm = self.x_proj(x_conv)
        dt, B_input, C = x_ssm[:, :, :1], x_ssm[:, :, 1:self.d_state+1], x_ssm[:, :, self.d_state+1:]
        dt = F.softplus(self.dt_proj(dt))
        A = self.A
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            dt_t, B_t, C_t, x_t = dt[:, t, :], B_input[:, t, :], C[:, t, :], x_conv[:, t, :]
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            y_t = (h * C_t.unsqueeze(1)).sum(-1) + self.D * x_t
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y))

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model), nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.ssm(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# ============ Linear Attention Hybrid Model ============
class Transformer_LinearAttn(nn.Module):
    """Linear Attention Hybrid: Mamba + Linear Attention (all O(n)).

    Architecture for 4 layers: [Mamba, Mamba, LinearAttn, Mamba]
    - SSM layers: O(n) sequential processing
    - Linear attention: O(n) global context (instead of O(n²) softmax attention)

    Testing hypothesis: Can linear attention provide similar benefits to
    standard attention while maintaining O(n) complexity throughout?
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        d_state = max(16, d_model // 16)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)

        # Build hybrid stack: linear attention every 3rd layer (like Hybrid)
        blocks = []
        for i in range(n_layers):
            if (i + 1) % 3 == 0:  # Layer 3, 6, 9... get linear attention
                blocks.append(LinearAttnBlock(d_model, n_heads))
            else:
                blocks.append(MambaBlock(d_model, d_state))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
''',

    # GQA-Mamba Hybrid (fast GQA + quality SSM)
    "Transformer_GQAMamba": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ GQA Block ============
class GQA(nn.Module):
    """Grouped Query Attention - fewer KV heads for speed"""
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
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        n_rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(x)

class GQABlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GQA(d_model, n_heads, n_kv_heads=2)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model), nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# ============ SSM Block ============
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            self.A.copy_(-torch.exp(torch.linspace(0, 4, d_state)).unsqueeze(0).expand(self.d_inner, -1))

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_main.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        x_ssm = self.x_proj(x_conv)
        dt, B_input, C = x_ssm[:, :, :1], x_ssm[:, :, 1:self.d_state+1], x_ssm[:, :, self.d_state+1:]
        dt = F.softplus(self.dt_proj(dt))
        A = self.A
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            dt_t, B_t, C_t, x_t = dt[:, t, :], B_input[:, t, :], C[:, t, :], x_conv[:, t, :]
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            y_t = (h * C_t.unsqueeze(1)).sum(-1) + self.D * x_t
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y))

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model), nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.ssm(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer_GQAMamba(nn.Module):
    """GQA-Mamba Hybrid: Fast GQA attention + quality SSM.

    Architecture: [GQA, Mamba, GQA, Mamba]
    - GQA: ~3.5x faster than MHA (fewer KV heads)
    - Mamba: Quality boost from SSM
    - Alternating pattern for balanced processing

    Hypothesis: Get close to MHA baseline speed while keeping quality gains.
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        d_state = max(16, d_model // 16)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        # [GQA, Mamba, GQA, Mamba]
        self.blocks = nn.ModuleList([
            GQABlock(d_model, n_heads),
            MambaBlock(d_model, d_state),
            GQABlock(d_model, n_heads),
            MambaBlock(d_model, d_state),
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
''',

    # Sandwich Architecture (attention bookends)
    "Transformer_Sandwich": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ MHA Block ============
class MHA(nn.Module):
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
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(x)

class MHABlock(nn.Module):
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
        x = x + self.ffn(self.ln2(x))
        return x

# ============ SSM Block ============
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            self.A.copy_(-torch.exp(torch.linspace(0, 4, d_state)).unsqueeze(0).expand(self.d_inner, -1))

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_main.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        x_ssm = self.x_proj(x_conv)
        dt, B_input, C = x_ssm[:, :, :1], x_ssm[:, :, 1:self.d_state+1], x_ssm[:, :, self.d_state+1:]
        dt = F.softplus(self.dt_proj(dt))
        A = self.A
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            dt_t, B_t, C_t, x_t = dt[:, t, :], B_input[:, t, :], C[:, t, :], x_conv[:, t, :]
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            y_t = (h * C_t.unsqueeze(1)).sum(-1) + self.D * x_t
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y))

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model), nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.ssm(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer_Sandwich(nn.Module):
    """Sandwich Architecture: Attention bookends with SSM filling.

    Architecture: [MHA, Mamba, Mamba, MHA]
    - First MHA: Establish global context
    - Middle Mamba layers: Efficient sequential processing
    - Final MHA: Global cleanup/refinement

    Hypothesis: Bookending with attention gives best of both worlds -
    global context at start AND end, with efficient SSM in middle.
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        d_state = max(16, d_model // 16)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        # [MHA, Mamba, Mamba, MHA]
        self.blocks = nn.ModuleList([
            MHABlock(d_model, n_heads),
            MambaBlock(d_model, d_state),
            MambaBlock(d_model, d_state),
            MHABlock(d_model, n_heads),
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
''',

    # Mamba 5:1 (push the ratio further)
    "Transformer_Mamba5to1": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ MHA Block ============
class MHA(nn.Module):
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
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(x)

class MHABlock(nn.Module):
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
        x = x + self.ffn(self.ln2(x))
        return x

# ============ SSM Block ============
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            self.A.copy_(-torch.exp(torch.linspace(0, 4, d_state)).unsqueeze(0).expand(self.d_inner, -1))

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_main.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        x_ssm = self.x_proj(x_conv)
        dt, B_input, C = x_ssm[:, :, :1], x_ssm[:, :, 1:self.d_state+1], x_ssm[:, :, self.d_state+1:]
        dt = F.softplus(self.dt_proj(dt))
        A = self.A
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            dt_t, B_t, C_t, x_t = dt[:, t, :], B_input[:, t, :], C[:, t, :], x_conv[:, t, :]
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            y_t = (h * C_t.unsqueeze(1)).sum(-1) + self.D * x_t
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y))

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model), nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.ssm(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer_Mamba5to1(nn.Module):
    """Mamba 5:1: Push the ratio even further.

    Architecture: [Mamba, Mamba, Mamba, Mamba, Mamba, MHA]
    - 5 Mamba layers for maximum SSM quality
    - 1 MHA at end for global cleanup

    Hypothesis: 4:1 beat 3:1. Will 5:1 beat 4:1?
    Testing if more Mamba continues to improve quality.
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        d_state = max(16, d_model // 16)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        # [M, M, M, M, M, MHA] - 6 layers total
        blocks = []
        for i in range(5):
            blocks.append(MambaBlock(d_model, d_state))
        blocks.append(MHABlock(d_model, n_heads))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
''',

    # GQA-Mamba Heavy (GQA start + 3 Mamba for quality)
    "Transformer_GQAMambaHeavy": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=2, dropout=0.1):
        super().__init__()
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = self.dropout(F.softmax(attn, dim=-1))
        return self.out((attn @ v).transpose(1, 2).reshape(B, N, C))

class GQABlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = GQA(d_model, n_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv-1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner)
        self.A = nn.Parameter(-torch.exp(torch.linspace(0, 4, d_state)).unsqueeze(0).expand(self.d_inner, -1).clone())
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)
        x_conv = F.silu(self.conv1d(x_main.transpose(1,2))[:,:,:L].transpose(1,2))
        x_ssm = self.x_proj(x_conv)
        dt = F.softplus(self.dt_proj(x_ssm[:,:,:1]))
        B_in, C = x_ssm[:,:,1:self.d_state+1], x_ssm[:,:,self.d_state+1:]
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(L):
            A_bar = torch.exp(dt[:,t,:].unsqueeze(-1) * self.A.unsqueeze(0))
            B_bar = dt[:,t,:].unsqueeze(-1) * B_in[:,t,:].unsqueeze(1)
            h = A_bar * h + B_bar * x_conv[:,t,:].unsqueeze(-1)
            outs.append((h * C[:,t,:].unsqueeze(1)).sum(-1) + self.D * x_conv[:,t,:])
        return self.dropout(self.out_proj(torch.stack(outs, 1) * F.silu(z)))

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.ssm(self.ln1(x))
        return x + self.ffn(self.ln2(x))

class Transformer_GQAMambaHeavy(nn.Module):
    """GQA-Mamba Heavy: Fast GQA start + 3 quality Mamba layers.
    Architecture: [GQA, Mamba, Mamba, Mamba]
    Hypothesis: GQA establishes fast global context, Mamba refines with quality.
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        d_state = max(16, d_model // 16)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        self.blocks = nn.ModuleList([GQABlock(d_model, n_heads)] + [MambaBlock(d_model, d_state) for _ in range(3)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks: x = block(x)
        return self.head(self.ln_f(x))
''',

    # MQA-Mamba (most KV-efficient attention + SSM)
    "Transformer_MQAMamba": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MQA(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.head_dim)
        self.v_proj = nn.Linear(d_model, self.head_dim)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, 1, self.head_dim).transpose(1, 2).repeat(1, self.n_heads, 1, 1)
        v = self.v_proj(x).reshape(B, N, 1, self.head_dim).transpose(1, 2).repeat(1, self.n_heads, 1, 1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = self.dropout(F.softmax(attn, dim=-1))
        return self.out((attn @ v).transpose(1, 2).reshape(B, N, C))

class MQABlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MQA(d_model, n_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv-1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner)
        self.A = nn.Parameter(-torch.exp(torch.linspace(0, 4, d_state)).unsqueeze(0).expand(self.d_inner, -1).clone())
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)
        x_conv = F.silu(self.conv1d(x_main.transpose(1,2))[:,:,:L].transpose(1,2))
        x_ssm = self.x_proj(x_conv)
        dt = F.softplus(self.dt_proj(x_ssm[:,:,:1]))
        B_in, C = x_ssm[:,:,1:self.d_state+1], x_ssm[:,:,self.d_state+1:]
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(L):
            A_bar = torch.exp(dt[:,t,:].unsqueeze(-1) * self.A.unsqueeze(0))
            B_bar = dt[:,t,:].unsqueeze(-1) * B_in[:,t,:].unsqueeze(1)
            h = A_bar * h + B_bar * x_conv[:,t,:].unsqueeze(-1)
            outs.append((h * C[:,t,:].unsqueeze(1)).sum(-1) + self.D * x_conv[:,t,:])
        return self.dropout(self.out_proj(torch.stack(outs, 1) * F.silu(z)))

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.ssm(self.ln1(x))
        return x + self.ffn(self.ln2(x))

class Transformer_MQAMamba(nn.Module):
    """MQA-Mamba: Most KV-efficient attention + SSM quality.
    Architecture: [MQA, Mamba, MQA, Mamba]
    Hypothesis: MQA even more efficient than GQA, test if quality holds.
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        d_state = max(16, d_model // 16)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        self.blocks = nn.ModuleList([MQABlock(d_model, n_heads), MambaBlock(d_model, d_state), MQABlock(d_model, n_heads), MambaBlock(d_model, d_state)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks: x = block(x)
        return self.head(self.ln_f(x))
''',

    # GQA-Mamba5 (GQA speed start + 5 Mamba quality)
    "Transformer_GQAMamba5": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=2, dropout=0.1):
        super().__init__()
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = self.dropout(F.softmax(attn, dim=-1))
        return self.out((attn @ v).transpose(1, 2).reshape(B, N, C))

class GQABlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = GQA(d_model, n_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv-1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner)
        self.A = nn.Parameter(-torch.exp(torch.linspace(0, 4, d_state)).unsqueeze(0).expand(self.d_inner, -1).clone())
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)
        x_conv = F.silu(self.conv1d(x_main.transpose(1,2))[:,:,:L].transpose(1,2))
        x_ssm = self.x_proj(x_conv)
        dt = F.softplus(self.dt_proj(x_ssm[:,:,:1]))
        B_in, C = x_ssm[:,:,1:self.d_state+1], x_ssm[:,:,self.d_state+1:]
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(L):
            A_bar = torch.exp(dt[:,t,:].unsqueeze(-1) * self.A.unsqueeze(0))
            B_bar = dt[:,t,:].unsqueeze(-1) * B_in[:,t,:].unsqueeze(1)
            h = A_bar * h + B_bar * x_conv[:,t,:].unsqueeze(-1)
            outs.append((h * C[:,t,:].unsqueeze(1)).sum(-1) + self.D * x_conv[:,t,:])
        return self.dropout(self.out_proj(torch.stack(outs, 1) * F.silu(z)))

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.ssm(self.ln1(x))
        return x + self.ffn(self.ln2(x))

class Transformer_GQAMamba5(nn.Module):
    """GQA-Mamba5: GQA speed + 5 Mamba quality layers.
    Architecture: [GQA, Mamba, Mamba, Mamba, Mamba, Mamba]
    Hypothesis: Fast GQA start, then max Mamba for quality push.
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        d_state = max(16, d_model // 16)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        self.blocks = nn.ModuleList([GQABlock(d_model, n_heads)] + [MambaBlock(d_model, d_state) for _ in range(5)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks: x = block(x)
        return self.head(self.ln_f(x))
''',

    # ============================================================
    # SPEED TIER: Fast GQA/MQA variants targeting <100s training
    # ============================================================

    # GQA3_MHA: 3 GQA + 1 MHA anchor (expected ~91s)
    "Transformer_GQA3_MHA": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    """Grouped Query Attention with causal masking"""
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
        n_rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(n_rep, dim=1).contiguous()
        v = v.repeat_interleave(n_rep, dim=1).contiguous()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        return self.out(x)

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
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(x)

class GQABlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GQA(d_model, n_heads, n_kv_heads=2)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class MHABlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer_GQA3_MHA(nn.Module):
    """GQA3_MHA: 3 fast GQA layers + 1 MHA anchor at end.
    Architecture: [GQA, GQA, GQA, MHA]
    Hypothesis: GQA speed with MHA quality anchor at end.
    Expected time: ~91s (3*14s + 49s)
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        self.blocks = nn.ModuleList([GQABlock(d_model, n_heads) for _ in range(3)] + [MHABlock(d_model, n_heads)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks: x = block(x)
        return self.head(self.ln_f(x))
''',

    # MQA3_MHA: 3 MQA + 1 MHA anchor (expected ~94s)
    "Transformer_MQA3_MHA": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MQA(nn.Module):
    """Multi-Query Attention with causal masking"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.head_dim)
        self.v_proj = nn.Linear(d_model, self.head_dim)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(x).reshape(B, N, 1, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(x).reshape(B, N, 1, self.head_dim).transpose(1, 2).contiguous()
        k = k.repeat(1, self.n_heads, 1, 1)
        v = v.repeat(1, self.n_heads, 1, 1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        return self.out(x)

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
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(x)

class MQABlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MQA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class MHABlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer_MQA3_MHA(nn.Module):
    """MQA3_MHA: 3 fast MQA layers + 1 MHA anchor at end.
    Architecture: [MQA, MQA, MQA, MHA]
    Hypothesis: MQA speed with MHA quality anchor at end.
    Expected time: ~94s (3*15s + 49s)
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        self.blocks = nn.ModuleList([MQABlock(d_model, n_heads) for _ in range(3)] + [MHABlock(d_model, n_heads)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks: x = block(x)
        return self.head(self.ln_f(x))
''',

    # GQA_MQA_Mix: Alternating GQA and MQA (expected ~58s)
    "Transformer_GQA_MQA_Mix": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    """Grouped Query Attention with causal masking"""
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
        n_rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(n_rep, dim=1).contiguous()
        v = v.repeat_interleave(n_rep, dim=1).contiguous()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        return self.out(x)

class MQA(nn.Module):
    """Multi-Query Attention with causal masking"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.head_dim)
        self.v_proj = nn.Linear(d_model, self.head_dim)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(x).reshape(B, N, 1, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(x).reshape(B, N, 1, self.head_dim).transpose(1, 2).contiguous()
        k = k.repeat(1, self.n_heads, 1, 1)
        v = v.repeat(1, self.n_heads, 1, 1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        return self.out(x)

class GQABlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GQA(d_model, n_heads, n_kv_heads=2)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class MQABlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MQA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer_GQA_MQA_Mix(nn.Module):
    """GQA_MQA_Mix: Alternating GQA and MQA layers.
    Architecture: [GQA, MQA, GQA, MQA]
    Hypothesis: Combining different KV-sharing strategies may complement each other.
    Expected time: ~58s (2*14s + 2*15s)
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        self.blocks = nn.ModuleList([GQABlock(d_model, n_heads), MQABlock(d_model, n_heads), GQABlock(d_model, n_heads), MQABlock(d_model, n_heads)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks: x = block(x)
        return self.head(self.ln_f(x))
''',

    # DeepGQA6: 6-layer pure GQA (expected ~84s)
    "Transformer_DeepGQA6": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    """Grouped Query Attention with causal masking"""
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
        n_rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(n_rep, dim=1).contiguous()
        v = v.repeat_interleave(n_rep, dim=1).contiguous()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        return self.out(x)

class GQABlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GQA(d_model, n_heads, n_kv_heads=2)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer_DeepGQA6(nn.Module):
    """DeepGQA6: 6-layer pure GQA for more depth.
    Architecture: [GQA, GQA, GQA, GQA, GQA, GQA]
    Hypothesis: More depth with fast layers may improve quality.
    Expected time: ~84s (6*14s)
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        self.blocks = nn.ModuleList([GQABlock(d_model, n_heads) for _ in range(6)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks: x = block(x)
        return self.head(self.ln_f(x))
''',

    # DeepGQA8: 8-layer pure GQA (expected ~112s)
    "Transformer_DeepGQA8": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    """Grouped Query Attention with causal masking"""
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
        n_rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(n_rep, dim=1).contiguous()
        v = v.repeat_interleave(n_rep, dim=1).contiguous()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        return self.out(x)

class GQABlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GQA(d_model, n_heads, n_kv_heads=2)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer_DeepGQA8(nn.Module):
    """DeepGQA8: 8-layer pure GQA for more depth.
    Architecture: [GQA x 8]
    Hypothesis: Testing if DeepGQA6 quality scales with more depth.
    Expected time: ~112s (8*14s)
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        self.blocks = nn.ModuleList([GQABlock(d_model, n_heads) for _ in range(8)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks: x = block(x)
        return self.head(self.ln_f(x))
''',

    # DeepGQA10: 10-layer pure GQA (expected ~140s)
    "Transformer_DeepGQA10": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    """Grouped Query Attention with causal masking"""
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
        n_rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(n_rep, dim=1).contiguous()
        v = v.repeat_interleave(n_rep, dim=1).contiguous()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        return self.out(x)

class GQABlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GQA(d_model, n_heads, n_kv_heads=2)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer_DeepGQA10(nn.Module):
    """DeepGQA10: 10-layer pure GQA for even more depth.
    Architecture: [GQA x 10]
    Hypothesis: Testing if DeepGQA quality continues to scale with 10 layers.
    Expected time: ~140s (10*14s)
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        self.blocks = nn.ModuleList([GQABlock(d_model, n_heads) for _ in range(10)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks: x = block(x)
        return self.head(self.ln_f(x))
''',

    # ============ LAYER SCALING COMPARISON (arcfusion-vho) ============
    # MHA variants for apples-to-apples comparison with DeepGQA at different depths

    "Transformer_MHA10": '''
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
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(x)

class MHABlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer_MHA10(nn.Module):
    """MHA10: 10-layer pure MHA for apples-to-apples comparison with DeepGQA10.
    Architecture: [MHA x 10]
    Comparing attention mechanism efficiency at equal depth.
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        self.blocks = nn.ModuleList([MHABlock(d_model, n_heads) for _ in range(10)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks: x = block(x)
        return self.head(self.ln_f(x))
''',

    "Transformer_MHA12": '''
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
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(x)

class MHABlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer_MHA12(nn.Module):
    """MHA12: 12-layer pure MHA for layer scaling comparison.
    Architecture: [MHA x 12]
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        self.blocks = nn.ModuleList([MHABlock(d_model, n_heads) for _ in range(12)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks: x = block(x)
        return self.head(self.ln_f(x))
''',

    "Transformer_MHA14": '''
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
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(x)

class MHABlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer_MHA14(nn.Module):
    """MHA14: 14-layer pure MHA for apples-to-apples comparison with DeepGQA14.
    Architecture: [MHA x 14]
    Comparing attention mechanism efficiency at equal depth.
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        self.blocks = nn.ModuleList([MHABlock(d_model, n_heads) for _ in range(14)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks: x = block(x)
        return self.head(self.ln_f(x))
''',

    "Transformer_DeepGQA12": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    """Grouped Query Attention with causal masking"""
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
        n_rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(n_rep, dim=1).contiguous()
        v = v.repeat_interleave(n_rep, dim=1).contiguous()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        return self.out(x)

class GQABlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GQA(d_model, n_heads, n_kv_heads=2)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer_DeepGQA12(nn.Module):
    """DeepGQA12: 12-layer pure GQA for layer scaling comparison.
    Architecture: [GQA x 12]
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        self.blocks = nn.ModuleList([GQABlock(d_model, n_heads) for _ in range(12)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks: x = block(x)
        return self.head(self.ln_f(x))
''',

    "Transformer_DeepGQA14": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    """Grouped Query Attention with causal masking"""
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
        n_rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(n_rep, dim=1).contiguous()
        v = v.repeat_interleave(n_rep, dim=1).contiguous()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        return self.out(x)

class GQABlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GQA(d_model, n_heads, n_kv_heads=2)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer_DeepGQA14(nn.Module):
    """DeepGQA14: 14-layer pure GQA for layer scaling comparison.
    Architecture: [GQA x 14]
    Testing if GQA efficiency advantage grows with depth.
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        self.blocks = nn.ModuleList([GQABlock(d_model, n_heads) for _ in range(14)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks: x = block(x)
        return self.head(self.ln_f(x))
''',

    "Transformer_DeepGQA18": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    """Grouped Query Attention with causal masking"""
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
        n_rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(n_rep, dim=1).contiguous()
        v = v.repeat_interleave(n_rep, dim=1).contiguous()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        return self.out(x)

class GQABlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GQA(d_model, n_heads, n_kv_heads=2)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer_DeepGQA18(nn.Module):
    """DeepGQA18: 18-layer pure GQA for maximum depth scaling.
    Architecture: [GQA x 18]
    Testing if GQA continues to scale at extreme depth.
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        self.blocks = nn.ModuleList([GQABlock(d_model, n_heads) for _ in range(18)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks: x = block(x)
        return self.head(self.ln_f(x))
''',

    "Transformer_MHA18": '''
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
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(x)

class MHABlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer_MHA18(nn.Module):
    """MHA18: 18-layer pure MHA for apples-to-apples comparison with DeepGQA18.
    Architecture: [MHA x 18]
    """
    def __init__(self, d_model=256, vocab_size=8000, n_layers=4, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(5000, d_model)
        self.blocks = nn.ModuleList([MHABlock(d_model, n_heads) for _ in range(18)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks: x = block(x)
        return self.head(self.ln_f(x))
''',
}


def config_hash(config: dict) -> str:
    """Generate a hash for config to match cached baselines."""
    import hashlib
    key_fields = ["d_model", "n_layers", "n_heads", "vocab_size", "seq_len", "max_steps"]
    content = "-".join(str(config.get(k, "")) for k in key_fields)
    return hashlib.sha256(content.encode()).hexdigest()[:8]


def result_to_training_run(result: dict, config: dict, baseline_run_id: str = "", model_code: str = "") -> TrainingRun:
    """Convert Modal result dict to TrainingRun dataclass."""
    # Local import to avoid Modal serialization issues
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "arcfusion"))
    from db import TrainingRun
    return TrainingRun(
        model_name=result["model_name"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        vocab_size=config["vocab_size"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        max_steps=config["max_steps"],
        seed=result.get("seed", config.get("seed", 42)),
        gpu_type=result.get("gpu", config.get("gpu", "A10G")),
        mixed_precision=result.get("mixed_precision", config.get("mixed_precision", True)),
        parameters=result.get("parameters", 0),
        final_train_loss=result.get("final_train_loss", 0.0),
        eval_loss=result.get("eval_loss", 0.0),
        perplexity=result.get("perplexity", 0.0),
        time_seconds=result.get("time_seconds", 0.0),
        success=result.get("success", False),
        error=result.get("error", ""),
        is_baseline=result.get("is_baseline", False),
        baseline_run_id=baseline_run_id,
        model_code=model_code,  # Store full model code for reproducibility
        # code_hash is auto-computed by TrainingRun dataclass
        notes=f"config_hash={config_hash(config)}, dataset={config.get('dataset', 'wikitext-2')}",
    )


def get_cached_baseline(db: ArcFusionDB, config: dict) -> TrainingRun | None:
    """Get cached baseline if config matches."""
    cfg_hash = config_hash(config)
    runs = db.list_training_runs(model_name=BASELINE_MODEL, baseline_only=True, success_only=True, limit=10)
    for run in runs:
        if f"config_hash={cfg_hash}" in run.notes:
            return run
    return None


def save_result_to_db(db: ArcFusionDB, result: dict, config: dict, baseline_run_id: str = "", model_code: str = "") -> str:
    """Save training result to database.

    Args:
        db: Database connection
        result: Training result dict from Modal
        config: Training config dict
        baseline_run_id: Optional baseline run ID for comparison
        model_code: Full Python code for model reproducibility (from MODELS dict)
    """
    run = result_to_training_run(result, config, baseline_run_id, model_code)

    # Calculate vs_baseline_pct if we have a baseline
    if baseline_run_id:
        baseline = db.get_training_run(baseline_run_id)
        if baseline and baseline.perplexity > 0:
            run.vs_baseline_pct = ((run.perplexity - baseline.perplexity) / baseline.perplexity) * 100

    return db.add_training_run(run)


def main():
    import sys
    # Defer db imports to avoid Modal serialization issues
    sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "arcfusion"))
    from db import ArcFusionDB, TrainingRun
    print("Starting main()...", flush=True)
    sys.stdout.flush()
    print("=" * 70, flush=True)
    print("FAIR COMPARISON - Same structure, different attention mechanisms", flush=True)
    print("=" * 70, flush=True)
    print(f"Config: {CONFIG['n_layers']} layers, d_model={CONFIG['d_model']}, {CONFIG['max_steps']} steps")
    print(f"GPU: {CONFIG['gpu']}, Mixed Precision: {CONFIG['mixed_precision']}")
    print(f"Baseline: {BASELINE_TARGET_RUNS} runs on {BASELINE_GPU}, seeds={BASELINE_SEEDS}")
    print()

    # Open DB connection
    db = ArcFusionDB("arcfusion.db")
    print(f"Database: arcfusion.db (training_runs: {db.stats().get('training_runs', 0)} existing)")

    results = {}
    cfg_hash = config_hash(CONFIG)

    # Check baseline status
    baseline_stats = db.get_baseline_stats(BASELINE_MODEL, cfg_hash)
    seeds_needed = db.get_baseline_seeds_needed(BASELINE_TARGET_RUNS, BASELINE_MODEL, cfg_hash)

    print(f"\n{'='*70}")
    print(f"BASELINE STATUS: {BASELINE_MODEL}")
    print("=" * 70)
    print(f"  Existing runs: {baseline_stats['n_runs']}/{BASELINE_TARGET_RUNS}")
    if baseline_stats['n_runs'] > 0:
        print(f"  Mean perplexity: {baseline_stats['mean_ppl']:.2f}")
        print(f"  Std deviation:   {baseline_stats['std_ppl']:.2f}")
        print(f"  Seeds run: {[r.seed for r in baseline_stats['runs']]}")
    if seeds_needed:
        print(f"  Seeds needed: {seeds_needed}")

    # Train any missing baseline runs
    if seeds_needed:
        print(f"\n{'='*70}")
        print(f"Training {len(seeds_needed)} BASELINE runs on {BASELINE_GPU}")
        print("=" * 70)

        baseline_code = MODELS[BASELINE_MODEL]
        for seed in seeds_needed:
            print(f"\n--- Baseline seed={seed} ---", flush=True)
            baseline_config = CONFIG.copy()
            baseline_config["seed"] = seed

            with app.run():
                baseline_result = train_model.remote(baseline_code, BASELINE_MODEL, baseline_config)

            baseline_result["is_baseline"] = True

            if baseline_result["success"]:
                print(f"  Perplexity: {baseline_result['perplexity']:.2f}")
                print(f"  Time: {baseline_result['time_seconds']:.1f}s")

                # Save to DB with model code for reproducibility
                run_id = save_result_to_db(db, baseline_result, baseline_config, model_code=baseline_code)
                print(f"  Saved to DB: {run_id}")
            else:
                print(f"  FAILED: {baseline_result['error']}")

        # Refresh baseline stats
        baseline_stats = db.get_baseline_stats(BASELINE_MODEL, cfg_hash)

    # Use mean perplexity for comparison
    baseline_ppl = baseline_stats['mean_ppl'] if baseline_stats['n_runs'] > 0 else None
    baseline_std = baseline_stats['std_ppl'] if baseline_stats['n_runs'] > 0 else 0

    if baseline_ppl:
        print(f"\n{'='*70}")
        print(f"USING CACHED BASELINE: {BASELINE_MODEL}")
        print("=" * 70)
        print(f"  N runs: {baseline_stats['n_runs']}")
        print(f"  Mean perplexity: {baseline_ppl:.2f} ± {baseline_std:.2f}")
        individual_ppls = [f"{r.perplexity:.2f}" for r in baseline_stats['runs']]
        print(f"  Individual: {individual_ppls}")

        # Add baseline to results (using mean values)
        import math
        results[BASELINE_MODEL] = {
            "success": True,
            "model_name": BASELINE_MODEL,
            "parameters": baseline_stats['runs'][0].parameters if baseline_stats['runs'] else 0,
            "eval_loss": math.log(baseline_ppl) if baseline_ppl > 0 else 0,
            "perplexity": baseline_ppl,
            "perplexity_std": baseline_std,
            "n_runs": baseline_stats['n_runs'],
            "is_baseline": True,
            "cached": True,
        }

    # Train other models on A10G
    for name, code in MODELS.items():
        if name == BASELINE_MODEL:
            continue  # Already handled

        print(f"\n{'='*70}")
        print(f"Training: {name} on {CONFIG['gpu']}")
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

            # Compare to baseline mean
            if baseline_ppl is not None:
                delta = result["perplexity"] - baseline_ppl
                pct = (delta / baseline_ppl) * 100
                # Check if outside 1 std dev
                significant = abs(delta) > baseline_std if baseline_std > 0 else True
                marker = " *" if significant else ""
                if delta < 0:
                    print(f"  vs Baseline: {delta:.2f} ({pct:+.1f}%) BETTER{marker}")
                else:
                    print(f"  vs Baseline: +{delta:.2f} ({pct:+.1f}%) worse{marker}")

            # Save to DB with model code for reproducibility
            run_id = save_result_to_db(db, result, CONFIG, model_code=code)
            print(f"  Saved to DB: {run_id}")
        else:
            print(f"\nFAILED: {result['error']}")

    # Summary with baseline comparison
    print("\n" + "=" * 70)
    print("FAIR COMPARISON RESULTS")
    print("=" * 70)
    if baseline_ppl:
        print(f"Baseline: {BASELINE_MODEL} mean={baseline_ppl:.2f} ± {baseline_std:.2f} (n={baseline_stats['n_runs']})")
    else:
        print(f"Baseline: {BASELINE_MODEL} - NO BASELINE DATA")
    print()
    print(f"{'Model':<20} {'Params':>12} {'Perplexity':>12} {'vs Baseline':>14} {'Time':>10}")
    print("-" * 78)

    sorted_results = sorted(
        [(k, v) for k, v in results.items() if v.get("success")],
        key=lambda x: x[1].get("eval_loss", float("inf"))
    )

    for name, r in sorted_results:
        if baseline_ppl and r.get("success"):
            ppl = r.get("perplexity", 0)
            delta = ppl - baseline_ppl
            pct = (delta / baseline_ppl) * 100
            if name == BASELINE_MODEL:
                vs_baseline = f"(baseline, n={r.get('n_runs', 1)})"
            else:
                vs_baseline = f"{pct:+.1f}%"
        else:
            vs_baseline = "N/A"
        params = r.get("parameters", 0)
        ppl = r.get("perplexity", 0)
        time_secs = r.get("time_seconds", 0)
        time_str = f"{time_secs:.0f}s" if time_secs else "N/A"
        print(f"{name:<20} {params:>12,} {ppl:>12.2f} {vs_baseline:>14} {time_str:>10}")

    # Save with metadata
    output = {
        "config": CONFIG,
        "baseline_model": BASELINE_MODEL,
        "baseline_stats": {
            "mean_ppl": baseline_ppl,
            "std_ppl": baseline_std,
            "n_runs": baseline_stats['n_runs'],
        },
        "results": results,
    }
    Path("experiments").mkdir(exist_ok=True)
    with open("experiments/fair_comparison_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to experiments/fair_comparison_results.json")

    # DB summary
    stats = db.stats()
    print(f"\nDatabase updated:")
    print(f"  Training runs: {stats.get('training_runs', 0)}")
    print(f"  Successful: {stats.get('successful_runs', 0)}")
    print(f"  Baselines: {stats.get('baseline_runs', 0)}")

    # Auto-generate insights from training results
    print(f"\n{'='*70}")
    print("GENERATING TRAINING INSIGHTS")
    print("=" * 70)
    from db import generate_training_insights
    new_insights = generate_training_insights(db)
    if new_insights:
        print(f"\nGenerated {len(new_insights)} new insights:")
        for insight in new_insights:
            print(f"  [{insight.category}] {insight.title}")
            if insight.description:
                print(f"      {insight.description[:80]}...")
    else:
        print("No new insights generated (insufficient data or all insights already exist)")

    db.close()


if __name__ == "__main__":
    main()
