#!/usr/bin/env python3
"""Train just Mamba with extended timeout (~32 min for 2000 steps)."""

import modal
import json
import math
import sys

modal.enable_output()

app = modal.App("arcfusion-mamba-only")

# Image with pre-cached WikiText-2
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "numpy", "datasets", "tiktoken")
    .run_commands(
        'python -c "from datasets import load_dataset; load_dataset(\'wikitext\', \'wikitext-2-raw-v1\', trust_remote_code=True)"'
    )
)

CONFIG = {
    "d_model": 256,
    "vocab_size": 50257,
    "n_layers": 4,
    "n_heads": 8,
    "seq_len": 128,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "max_steps": 2000,
    "eval_interval": 200,
    "mixed_precision": True,
    "gpu": "A100",
    "dataset": "wikitext-2",
    "seed": 42,
}

# Mamba model code - SSM attention
MAMBA_CODE = '''
class SSMBlock(nn.Module):
    """Selective State Space Model block - O(n) complexity."""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Input projection (expand to 2x for gating)
        self.in_proj = nn.Linear(d_model, d_model * 2)

        # SSM parameters
        self.A = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.dt_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape

        # Input projection and gate split
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        # SSM computation (simplified selective scan)
        dt = torch.nn.functional.softplus(self.dt_proj(x_in))  # [B, L, D]
        B_t = self.B_proj(x_in)  # [B, L, d_state]
        C_t = self.C_proj(x_in)  # [B, L, d_state]

        # Discretized state space (sequential scan)
        y = torch.zeros_like(x_in)
        h = torch.zeros(B, D, self.d_state, device=x.device)  # Hidden state

        A_bar = -torch.exp(self.A)  # Ensure stability

        for t in range(L):
            dt_t = dt[:, t:t+1, :]  # [B, 1, D]
            B_t_t = B_t[:, t, :]    # [B, d_state]
            C_t_t = C_t[:, t, :]    # [B, d_state]
            x_t = x_in[:, t, :]     # [B, D]

            # Discretize: A_d = exp(A * dt), B_d = dt * B
            dA = torch.exp(A_bar * dt_t.transpose(-1, -2))  # [B, D, d_state]
            dB = dt_t.transpose(-1, -2) * B_t_t.unsqueeze(1)  # [B, D, d_state]

            # State update: h = A_d * h + B_d * x
            h = h * dA + dB * x_t.unsqueeze(-1)

            # Output: y = C * h
            y_t = (h * C_t_t.unsqueeze(1)).sum(-1)  # [B, D]
            y[:, t, :] = y_t

        # Gate and output
        out = y * torch.nn.functional.silu(z)
        return self.ln(x + self.out_proj(out))

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.ssm = SSMBlock(d_model, d_state)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ssm(x)
        return x + self.ffn(self.ln(x))

class Attention(nn.Module):
    def __init__(self, d_model, n_heads, d_state=16):
        super().__init__()
        self.mamba = MambaBlock(d_model, d_state)

    def forward(self, x, mask=None):
        return self.mamba(x)
'''


@app.function(image=image, gpu="A100", timeout=2400)  # 40 min timeout
def train_mamba(config: dict) -> dict:
    """Train Mamba model for 2000 steps."""
    import time
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from datasets import load_dataset
    import tiktoken

    use_amp = config.get("mixed_precision", True)
    seed = config.get("seed", 42)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    result = {
        "success": False,
        "model_name": "Transformer_Mamba",
        "final_train_loss": float("inf"),
        "eval_loss": float("inf"),
        "perplexity": float("inf"),
        "steps": 0,
        "time_seconds": 0,
        "parameters": 0,
        "error": None,
        "gpu": "A100",
        "mixed_precision": use_amp,
        "dataset": "wikitext-2",
        "seed": seed,
        "is_baseline": False,
    }

    try:
        # Load data (should be instant from cache)
        print("Loading WikiText-2 (from cache)...", flush=True)
        data_start = time.time()
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        enc = tiktoken.get_encoding("gpt2")
        data_time = time.time() - data_start
        print(f"Data loaded in {data_time:.2f}s", flush=True)

        # Tokenize
        def tokenize_split(split_name):
            texts = [t for t in ds[split_name]["text"] if t.strip()]
            all_tokens = []
            for text in texts:
                all_tokens.extend(enc.encode(text))
            return all_tokens

        train_tokens = tokenize_split("train")
        val_tokens = tokenize_split("validation")
        print(f"Train tokens: {len(train_tokens):,}, Val tokens: {len(val_tokens):,}", flush=True)

        class TokenDataset(Dataset):
            def __init__(self, tokens, seq_len):
                self.tokens = torch.tensor(tokens, dtype=torch.long)
                self.seq_len = seq_len

            def __len__(self):
                return max(0, len(self.tokens) - self.seq_len - 1)

            def __getitem__(self, idx):
                chunk = self.tokens[idx : idx + self.seq_len + 1]
                return chunk[:-1], chunk[1:]

        seq_len = config["seq_len"]
        batch_size = config["batch_size"]
        train_loader = DataLoader(
            TokenDataset(train_tokens, seq_len),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            TokenDataset(val_tokens, seq_len),
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )

        # Build model
        d_model = config["d_model"]
        n_layers = config["n_layers"]
        n_heads = config["n_heads"]
        vocab_size = config["vocab_size"]

        # Execute Mamba code to get Attention class
        ns = {"nn": nn, "torch": torch, "math": math}
        exec(MAMBA_CODE, ns)
        Attention = ns["Attention"]

        class TransformerBlock(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                self.attn = Attention(d_model, n_heads)
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(0.1),
                )
                self.ln1 = nn.LayerNorm(d_model)
                self.ln2 = nn.LayerNorm(d_model)

            def forward(self, x):
                x = x + self.attn(self.ln1(x))
                x = x + self.ffn(self.ln2(x))
                return x

        class TransformerLM(nn.Module):
            def __init__(self, vocab_size, d_model, n_layers, n_heads, max_seq_len=5000):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, d_model)
                self.pos = nn.Embedding(max_seq_len, d_model)
                self.blocks = nn.ModuleList(
                    [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
                )
                self.ln_f = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, vocab_size)

            def forward(self, x):
                B, T = x.shape
                tok_emb = self.embed(x)
                pos_emb = self.pos(torch.arange(T, device=x.device))
                x = tok_emb + pos_emb
                for block in self.blocks:
                    x = block(x)
                x = self.ln_f(x)
                return self.head(x)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TransformerLM(vocab_size, d_model, n_layers, n_heads).to(device)
        result["parameters"] = sum(p.numel() for p in model.parameters())
        print(f"Model params: {result['parameters']:,}", flush=True)

        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler() if use_amp else None

        # Training loop
        max_steps = config["max_steps"]
        eval_interval = config["eval_interval"]
        model.train()
        train_iter = iter(train_loader)

        print(f"Starting training for {max_steps} steps...", flush=True)
        train_start = time.time()
        step = 0

        while step < max_steps:
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                loss.backward()
                optimizer.step()

            step += 1
            result["final_train_loss"] = loss.item()

            if step % 100 == 0:
                elapsed = time.time() - train_start
                print(f"Step {step}/{max_steps}, loss={loss.item():.4f}, time={elapsed:.1f}s", flush=True)

        result["steps"] = step
        result["time_seconds"] = time.time() - train_start
        print(f"Training done in {result['time_seconds']:.1f}s", flush=True)

        # Evaluation
        print("Running evaluation...", flush=True)
        model.eval()
        total_loss = 0
        num_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(x)
                        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                else:
                    logits = model(x)
                    loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                total_loss += loss.item()
                num_batches += 1
                if num_batches >= 50:  # Limit eval batches
                    break

        result["eval_loss"] = total_loss / num_batches
        result["perplexity"] = math.exp(min(result["eval_loss"], 20))
        result["success"] = True
        print(f"Eval loss: {result['eval_loss']:.4f}, Perplexity: {result['perplexity']:.2f}", flush=True)

    except Exception as e:
        import traceback
        result["error"] = f"{e}\n{traceback.format_exc()}"
        print(f"ERROR: {result['error']}", flush=True)

    return result


def main():
    print("=" * 60)
    print("MAMBA-ONLY TRAINING (2000 steps, ~32 minutes)")
    print("=" * 60)
    print(f"Config: {CONFIG['n_layers']} layers, d_model={CONFIG['d_model']}")
    print()

    with app.run():
        result = train_mamba.remote(CONFIG)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Success: {result['success']}")
    print(f"Parameters: {result['parameters']:,}")
    print(f"Final train loss: {result['final_train_loss']:.4f}")
    print(f"Eval loss: {result['eval_loss']:.4f}")
    print(f"Perplexity: {result['perplexity']:.2f}")
    print(f"Training time: {result['time_seconds']:.1f}s")
    if result["error"]:
        print(f"Error: {result['error'][:500]}")

    # Save result
    print()
    print("Saving to experiments/mamba_result.json...")
    import json
    with open("experiments/mamba_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Done!")


if __name__ == "__main__":
    main()
