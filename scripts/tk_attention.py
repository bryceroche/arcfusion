"""
ThunderKittens Attention Integration for ArcFusion

This module provides drop-in replacements for our attention implementations
using ThunderKittens CUDA kernels for improved performance.

Requirements:
- CUDA 12.3+ (12.6 recommended)
- GCC 11+ (C++20 support)
- ThunderKittens installed: pip install thunderkittens (or build from source)

Usage:
    # Replace standard attention with TK attention
    from tk_attention import TKMultiHeadAttention, TKGroupedQueryAttention

    # Drop-in replacement - same interface
    attn = TKMultiHeadAttention(d_model=512, n_heads=8)

References:
- Paper: https://arxiv.org/abs/2410.20399 (ICLR 2025)
- Repo: https://github.com/HazyResearch/ThunderKittens
- Training: https://github.com/HazyResearch/train-tk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Flag to track if TK is available
TK_AVAILABLE = False
tk_attention_fn = None

try:
    # Try to import ThunderKittens attention kernel
    # This will be available once TK is installed in Modal
    from thunderkittens import attention as tk_attn
    TK_AVAILABLE = True
    tk_attention_fn = tk_attn
    print("✓ ThunderKittens attention kernels loaded")
except ImportError:
    print("⚠ ThunderKittens not available, falling back to PyTorch SDPA")


def get_attention_fn(use_tk: bool = True):
    """Get the best available attention function.

    Args:
        use_tk: If True and TK is available, use TK. Otherwise fallback to SDPA.

    Returns:
        Attention function that takes (q, k, v) and returns output
    """
    if use_tk and TK_AVAILABLE:
        return tk_attention_fn
    else:
        # Fallback to PyTorch's scaled_dot_product_attention (uses FlashAttention if available)
        return F.scaled_dot_product_attention


class TKMultiHeadAttention(nn.Module):
    """Multi-Head Attention with ThunderKittens backend.

    Drop-in replacement for standard MHA. Uses TK kernels when available,
    falls back to PyTorch SDPA otherwise.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1,
                 use_tk: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_tk = use_tk and TK_AVAILABLE

        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply attention
        if self.use_tk and TK_AVAILABLE:
            # ThunderKittens attention
            # TK expects: (B, H, N, D) format - same as our q, k, v
            out = tk_attention_fn(q, k, v, causal=True)
        else:
            # Fallback to PyTorch SDPA
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True if mask is None else False
            )

        # Reshape and project output
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out


class TKGroupedQueryAttention(nn.Module):
    """Grouped Query Attention with ThunderKittens backend.

    GQA uses fewer KV heads than query heads for memory efficiency.
    Uses TK kernels when available.
    """

    def __init__(self, d_model: int, n_heads: int = 8, n_kv_heads: int = 2,
                 dropout: float = 0.1, use_tk: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.kv_dim = self.head_dim * n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.use_tk = use_tk and TK_AVAILABLE

        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.kv_dim)
        self.v_proj = nn.Linear(d_model, self.kv_dim)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape

        # Project Q, K, V
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV heads to match Q heads (repeat for each group)
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        # Apply attention
        if self.use_tk and TK_AVAILABLE:
            out = tk_attention_fn(q, k, v, causal=True)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True if mask is None else False
            )

        # Reshape and project output
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out


class TKLinearAttention(nn.Module):
    """Linear Attention with ThunderKittens backend.

    O(n) attention using feature maps instead of softmax.
    TK provides 14x speedup on Based linear attention.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1,
                 use_tk: bool = True, feature_map: str = "elu"):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_tk = use_tk and TK_AVAILABLE
        self.feature_map = feature_map

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature map for linear attention."""
        if self.feature_map == "elu":
            return F.elu(x) + 1
        elif self.feature_map == "relu":
            return F.relu(x)
        else:
            return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape

        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_tk and TK_AVAILABLE:
            # TODO: Use TK's Based or Hedgehog linear attention
            # For now, fall back to manual implementation
            try:
                from thunderkittens import based_attention
                out = based_attention(q, k, v)
            except ImportError:
                out = self._linear_attention_fallback(q, k, v)
        else:
            out = self._linear_attention_fallback(q, k, v)

        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out

    def _linear_attention_fallback(self, q, k, v):
        """Fallback O(n) linear attention implementation."""
        # Apply feature maps
        q = self._feature_map(q)
        k = self._feature_map(k)

        # Linear attention: O(n) complexity
        # out = (Q @ (K^T @ V)) / (Q @ K^T @ 1)
        kv = torch.einsum('bhnd,bhnm->bhdm', k, v)  # (B, H, D, D)
        qkv = torch.einsum('bhnd,bhdm->bhnm', q, kv)  # (B, H, N, D)

        # Normalizer
        k_sum = k.sum(dim=2, keepdim=True)  # (B, H, 1, D)
        normalizer = torch.einsum('bhnd,bhkd->bhnk', q, k_sum).clamp(min=1e-6)

        out = qkv / normalizer
        return out


# Modal image builder for TK environment
def get_tk_modal_image():
    """Get Modal image configured for ThunderKittens.

    Returns a Modal image with CUDA 12.6 and ThunderKittens installed.
    """
    import modal

    # Base image with CUDA 12.6
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.6.0-devel-ubuntu22.04",
            add_python="3.11"
        )
        # Install build dependencies
        .apt_install("git", "gcc-11", "g++-11", "make")
        .run_commands(
            "update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100",
            "update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100"
        )
        # Install Python dependencies
        .pip_install(
            "torch>=2.0",
            "numpy",
            "datasets",
            "tiktoken",
            "ninja"  # For faster compilation
        )
        # Clone and install ThunderKittens
        .run_commands(
            "git clone https://github.com/HazyResearch/ThunderKittens.git /opt/thunderkittens",
            "cd /opt/thunderkittens && python setup.py install"
        )
        .env({"THUNDERKITTENS_ROOT": "/opt/thunderkittens"})
    )

    return image


# Benchmark utility
def benchmark_attention(batch_size=4, seq_len=1024, d_model=512, n_heads=8,
                        iterations=100, warmup=10):
    """Benchmark TK vs standard attention.

    Returns dict with timing results.
    """
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create test input
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Standard attention (SDPA)
    std_attn = TKMultiHeadAttention(d_model, n_heads, use_tk=False).to(device)

    # TK attention
    tk_attn = TKMultiHeadAttention(d_model, n_heads, use_tk=True).to(device)

    results = {}

    for name, module in [("SDPA", std_attn), ("TK", tk_attn)]:
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = module(x)

        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            with torch.no_grad():
                _ = module(x)

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        results[name] = {
            "total_ms": elapsed * 1000,
            "per_iter_ms": elapsed * 1000 / iterations,
            "throughput": iterations / elapsed
        }

    # Compute speedup
    if "SDPA" in results and "TK" in results:
        results["speedup"] = results["SDPA"]["per_iter_ms"] / results["TK"]["per_iter_ms"]

    return results


if __name__ == "__main__":
    # Test the module
    print("Testing TK Attention Module")
    print(f"TK Available: {TK_AVAILABLE}")

    # Test with random input
    x = torch.randn(2, 128, 512)

    print("\nTesting TKMultiHeadAttention...")
    mha = TKMultiHeadAttention(512, 8, use_tk=False)  # Force SDPA for testing
    out = mha(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")

    print("\nTesting TKGroupedQueryAttention...")
    gqa = TKGroupedQueryAttention(512, 8, n_kv_heads=2, use_tk=False)
    out = gqa(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")

    print("\nTesting TKLinearAttention...")
    lin = TKLinearAttention(512, 8, use_tk=False)
    out = lin(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")

    print("\n✓ All attention modules working")
