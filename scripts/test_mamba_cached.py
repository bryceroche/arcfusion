#!/usr/bin/env python3
"""Test Mamba training with cached WikiText-2 data."""

import modal

modal.enable_output()

app = modal.App("arcfusion-mamba-cache-test")

# Same image as cloud_train_fair.py - with pre-cached data
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "numpy", "datasets", "tiktoken")
    .run_commands(
        'python -c "from datasets import load_dataset; load_dataset(\'wikitext\', \'wikitext-2-raw-v1\', trust_remote_code=True)"'
    )
)


@app.function(image=image, gpu="A100", timeout=1200)
def test_mamba_training():
    """Test Mamba training with cached data - quick 100 step run."""
    import time
    import math
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from datasets import load_dataset
    import tiktoken

    result = {"success": False, "data_load_time": 0, "training_time": 0, "error": None}

    try:
        # Time data loading
        data_start = time.time()
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        enc = tiktoken.get_encoding("gpt2")
        result["data_load_time"] = time.time() - data_start

        # Tokenize
        def tokenize_split(split_name):
            texts = [t for t in ds[split_name]["text"] if t.strip()]
            all_tokens = []
            for text in texts:
                all_tokens.extend(enc.encode(text))
            return all_tokens

        train_tokens = tokenize_split("train")

        class TokenDataset(Dataset):
            def __init__(self, tokens, seq_len):
                self.tokens = torch.tensor(tokens, dtype=torch.long)
                self.seq_len = seq_len

            def __len__(self):
                return max(0, len(self.tokens) - self.seq_len - 1)

            def __getitem__(self, idx):
                chunk = self.tokens[idx : idx + self.seq_len + 1]
                return chunk[:-1], chunk[1:]

        train_loader = DataLoader(
            TokenDataset(train_tokens, 128), batch_size=32, shuffle=True, drop_last=True
        )

        # Simple Mamba-like model (SSM block) for testing
        class SSMBlock(nn.Module):
            def __init__(self, d_model, d_state=16):
                super().__init__()
                self.d_model = d_model
                self.d_state = d_state
                self.in_proj = nn.Linear(d_model, d_model * 2)
                self.A = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
                self.B_proj = nn.Linear(d_model, d_state)
                self.C_proj = nn.Linear(d_model, d_state)
                self.dt_proj = nn.Linear(d_model, d_model)
                self.out_proj = nn.Linear(d_model, d_model)
                self.ln = nn.LayerNorm(d_model)

            def forward(self, x):
                B, L, D = x.shape
                xz = self.in_proj(x)
                x_in, z = xz.chunk(2, dim=-1)
                dt = torch.nn.functional.softplus(self.dt_proj(x_in))
                B_t = self.B_proj(x_in)
                C_t = self.C_proj(x_in)
                y = torch.zeros_like(x_in)
                h = torch.zeros(B, D, self.d_state, device=x.device)
                A_bar = -torch.exp(self.A)
                for t in range(L):
                    dt_t = dt[:, t : t + 1, :]
                    B_t_t = B_t[:, t, :]
                    C_t_t = C_t[:, t, :]
                    x_t = x_in[:, t, :]
                    dA = torch.exp(A_bar * dt_t.transpose(-1, -2))
                    dB = dt_t.transpose(-1, -2) * B_t_t.unsqueeze(1)
                    h = h * dA + dB * x_t.unsqueeze(-1)
                    y_t = (h * C_t_t.unsqueeze(1)).sum(-1)
                    y[:, t, :] = y_t
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
                    nn.Dropout(0.1),
                )
                self.ln = nn.LayerNorm(d_model)

            def forward(self, x):
                x = self.ssm(x)
                return x + self.ffn(self.ln(x))

        class MambaLM(nn.Module):
            def __init__(self, vocab_size, d_model=256, n_layers=4):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, d_model)
                self.pos = nn.Embedding(5000, d_model)
                self.blocks = nn.ModuleList(
                    [MambaBlock(d_model, d_state=16) for _ in range(n_layers)]
                )
                self.ln_f = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, vocab_size)

            def forward(self, x):
                B, T = x.shape
                x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
                for block in self.blocks:
                    x = block(x)
                return self.head(self.ln_f(x))

        # Training
        device = "cuda"
        model = MambaLM(enc.n_vocab).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        criterion = nn.CrossEntropyLoss()

        train_start = time.time()
        model.train()
        step = 0
        max_steps = 2000  # Full training
        train_iter = iter(train_loader)

        while step < max_steps:
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.reshape(-1, enc.n_vocab), y.reshape(-1))
            loss.backward()
            optimizer.step()
            step += 1

        result["training_time"] = time.time() - train_start
        result["final_loss"] = loss.item()
        result["perplexity"] = math.exp(min(loss.item(), 20))
        result["success"] = True

    except Exception as e:
        import traceback

        result["error"] = f"{e}\n{traceback.format_exc()}"

    return result


def main():
    print("Testing Mamba with cached WikiText-2 data...")
    with app.run():
        result = test_mamba_training.remote()

    print(f"Success: {result['success']}")
    print(f"Data load time: {result['data_load_time']:.2f}s (should be <5s if cached)")
    print(f"Training time: {result['training_time']:.2f}s")
    if result["success"]:
        print(f"Final loss: {result['final_loss']:.4f}")
        print(f"Perplexity: {result['perplexity']:.2f}")
    if result["error"]:
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
