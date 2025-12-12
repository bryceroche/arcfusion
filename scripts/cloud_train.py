#!/usr/bin/env python3
"""
Standalone cloud training script for ArcFusion architectures.
Run with Python 3.11: .venv-modal/bin/python scripts/cloud_train.py
"""

import modal
import time
import json
from pathlib import Path

# Modal app setup
app = modal.App("arcfusion-trainer")

# GPU image with PyTorch
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "numpy")
)

# Training configuration
DEFAULT_CONFIG = {
    "d_model": 256,
    "vocab_size": 8000,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "max_steps": 5000,
    "eval_interval": 100,
}


@app.function(image=image, gpu="T4", timeout=600)
def train_on_gpu(code: str, model_name: str, config: dict) -> dict:
    """Train a model on Modal GPU."""
    import time
    import math
    import torch
    import torch.nn as nn

    result = {
        "success": False,
        "model_name": model_name,
        "final_loss": float("inf"),
        "eval_loss": float("inf"),
        "perplexity": float("inf"),
        "steps_completed": 0,
        "training_time_seconds": 0.0,
        "num_parameters": 0,
        "error": None,
        "loss_history": [],
    }

    start_time = time.time()

    try:
        # Execute the generated code
        namespace = {}
        exec(code, namespace)

        # Get the model class
        model_class = namespace.get(model_name)
        if model_class is None:
            result["error"] = f"Model class '{model_name}' not found"
            return result

        # Instantiate model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training on: {device}")

        model = model_class(
            d_model=config["d_model"],
            vocab_size=config["vocab_size"],
        ).to(device)

        # Count parameters
        result["num_parameters"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"Model parameters: {result['num_parameters']:,}")

        # Training setup
        vocab_size = config["vocab_size"]
        batch_size = config["batch_size"]
        seq_len = 64

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"]
        )
        criterion = nn.CrossEntropyLoss()

        # Training loop
        model.train()
        max_steps = config["max_steps"]
        eval_interval = config["eval_interval"]

        for step in range(max_steps):
            # Generate random batch
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]

            # Forward pass
            logits = model(inputs)

            # Compute loss
            loss = criterion(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Record loss
            loss_val = loss.item()
            if step % eval_interval == 0:
                result["loss_history"].append({
                    "step": step,
                    "loss": loss_val,
                })
                print(f"Step {step}: loss={loss_val:.4f}")

            result["steps_completed"] = step + 1
            result["final_loss"] = loss_val

        # Evaluation pass
        model.eval()
        eval_losses = []
        with torch.no_grad():
            for _ in range(10):
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                logits = model(inputs)
                loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
                eval_losses.append(loss.item())

        result["eval_loss"] = sum(eval_losses) / len(eval_losses)

        # Compute perplexity
        if result["eval_loss"] < 20:
            result["perplexity"] = math.exp(result["eval_loss"])
        else:
            result["perplexity"] = float("inf")

        result["success"] = True

    except Exception as e:
        import traceback
        result["error"] = f"{str(e)}\n{traceback.format_exc()}"

    result["training_time_seconds"] = time.time() - start_time
    return result


# Model code templates - generated from our best architectures
MODELS = {
    "LLaMoE": '''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, d_model=512, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = kwargs.get('vocab_size', 32000)
        self.max_len = kwargs.get('max_len', 5000)
        self.token_emb = nn.Embedding(self.vocab_size, d_model)
        self.pos_emb = nn.Embedding(self.max_len, d_model)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))

    def forward(self, x, **kwargs):
        if x.dim() == 3:
            return x
        B, T = x.shape
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        return self.dropout(tok + pos)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model=512, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_heads = kwargs.get('n_heads', 8)
        self.head_dim = d_model // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)

class SoftmaxOutput(nn.Module):
    def __init__(self, d_model=512, **kwargs):
        super().__init__()
        self.vocab_size = kwargs.get('vocab_size', 32000)
        self.proj = nn.Linear(d_model, self.vocab_size)

    def forward(self, x, **kwargs):
        return self.proj(x)

class LLaMoE(nn.Module):
    def __init__(self, d_model=512, vocab_size=32000, **kwargs):
        super().__init__()
        self.rotary_embedding = RotaryEmbedding(d_model=d_model, vocab_size=vocab_size)
        self.grouped_query_attention = GroupedQueryAttention(d_model=d_model)
        self.softmax_output = SoftmaxOutput(d_model=d_model, vocab_size=vocab_size)

    def forward(self, x, **kwargs):
        x = self.rotary_embedding(x)
        x = self.grouped_query_attention(x)
        x = self.softmax_output(x)
        return x
''',

    "MambaFormer": '''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class InputEmbedding(nn.Module):
    def __init__(self, d_model=512, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = kwargs.get('vocab_size', 32000)
        self.max_len = kwargs.get('max_len', 5000)
        self.token_emb = nn.Embedding(self.vocab_size, d_model)
        self.pos_emb = nn.Embedding(self.max_len, d_model)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))

    def forward(self, x, **kwargs):
        if x.dim() == 3:
            return x
        B, T = x.shape
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        return self.dropout(tok + pos)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_heads = kwargs.get('n_heads', 8)
        self.head_dim = d_model // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)

class SoftmaxOutput(nn.Module):
    def __init__(self, d_model=512, **kwargs):
        super().__init__()
        self.vocab_size = kwargs.get('vocab_size', 32000)
        self.proj = nn.Linear(d_model, self.vocab_size)

    def forward(self, x, **kwargs):
        return self.proj(x)

class MambaFormer(nn.Module):
    def __init__(self, d_model=512, vocab_size=32000, **kwargs):
        super().__init__()
        self.input_embedding = InputEmbedding(d_model=d_model, vocab_size=vocab_size)
        self.multi_head_attention = MultiHeadAttention(d_model=d_model)
        self.softmax_output = SoftmaxOutput(d_model=d_model, vocab_size=vocab_size)

    def forward(self, x, **kwargs):
        x = self.input_embedding(x)
        x = self.multi_head_attention(x)
        x = self.softmax_output(x)
        return x
''',

    "Transformer": '''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class InputEmbedding(nn.Module):
    def __init__(self, d_model=512, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = kwargs.get('vocab_size', 32000)
        self.max_len = kwargs.get('max_len', 5000)
        self.token_emb = nn.Embedding(self.vocab_size, d_model)
        self.pos_emb = nn.Embedding(self.max_len, d_model)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))

    def forward(self, x, **kwargs):
        if x.dim() == 3:
            return x
        B, T = x.shape
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        return self.dropout(tok + pos)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_heads = kwargs.get('n_heads', 8)
        self.head_dim = d_model // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, d_model=512, **kwargs):
        super().__init__()
        self.d_ff = kwargs.get('d_ff', d_model * 4)
        self.fc1 = nn.Linear(d_model, self.d_ff)
        self.fc2 = nn.Linear(self.d_ff, d_model)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))

    def forward(self, x, **kwargs):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, **kwargs):
        super().__init__()
        self.attn = MultiHeadAttention(d_model=d_model, **kwargs)
        self.ff = FeedForward(d_model=d_model, **kwargs)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, **kwargs):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class SoftmaxOutput(nn.Module):
    def __init__(self, d_model=512, **kwargs):
        super().__init__()
        self.vocab_size = kwargs.get('vocab_size', 32000)
        self.proj = nn.Linear(d_model, self.vocab_size)

    def forward(self, x, **kwargs):
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, d_model=512, vocab_size=32000, n_layers=4, **kwargs):
        super().__init__()
        self.embedding = InputEmbedding(d_model=d_model, vocab_size=vocab_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, **kwargs) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.output = SoftmaxOutput(d_model=d_model, vocab_size=vocab_size)

    def forward(self, x, **kwargs):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.output(x)
        return x
''',
}


def main():
    """Run cloud training on all models."""
    print("=" * 70)
    print("ArcFusion Cloud Training - Modal GPU (T4)")
    print("=" * 70)

    results = {}

    for model_name, code in MODELS.items():
        print(f"\n{'='*70}")
        print(f"Training: {model_name}")
        print("=" * 70)

        with app.run():
            result = train_on_gpu.remote(code, model_name, DEFAULT_CONFIG)

        results[model_name] = result

        if result["success"]:
            print(f"\n{model_name} Results:")
            print(f"  Parameters: {result['num_parameters']:,}")
            print(f"  Final Loss: {result['final_loss']:.4f}")
            print(f"  Eval Loss: {result['eval_loss']:.4f}")
            print(f"  Perplexity: {result['perplexity']:.2f}")
            print(f"  Time: {result['training_time_seconds']:.1f}s")
        else:
            print(f"\n{model_name} FAILED: {result['error']}")

    # Summary
    print("\n" + "=" * 70)
    print("CLOUD TRAINING SUMMARY (5000 steps, T4 GPU)")
    print("=" * 70)
    print(f"{'Model':<20} {'Params':>12} {'Eval Loss':>12} {'Perplexity':>12}")
    print("-" * 70)

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get("perplexity", float("inf"))
    )

    for model_name, result in sorted_results:
        if result["success"]:
            print(f"{model_name:<20} {result['num_parameters']:>12,} {result['eval_loss']:>12.4f} {result['perplexity']:>12.2f}")
        else:
            print(f"{model_name:<20} {'FAILED':>12}")

    # Save results
    output_path = Path("experiments/cloud_training_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
