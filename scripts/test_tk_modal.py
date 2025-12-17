"""
ThunderKittens Modal Benchmark

Test and benchmark ThunderKittens attention kernels on Modal A100 GPUs.

Usage:
    # Deploy and run benchmark
    modal run scripts/test_tk_modal.py

    # Just check if TK builds
    modal run scripts/test_tk_modal.py::check_tk_install
"""

import modal
import time

app = modal.App("arcfusion-tk-benchmark")

# Modal image with CUDA 12.4 and ThunderKittens built from source
# Note: TK requires CUDA 12.3+, we use 12.4 for compatibility
tk_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    # Install build dependencies
    .apt_install("git", "gcc-11", "g++-11", "make", "ninja-build", "cmake")
    .run_commands(
        "update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100",
        "update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100"
    )
    # Install Python dependencies first
    .pip_install(
        "torch==2.4.0",
        "numpy",
        "ninja",
        "triton>=2.0",
        "packaging"
    )
    # Set environment
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        "TORCH_CUDA_ARCH_LIST": "8.0"  # A100 architecture
    })
    # Clone and build ThunderKittens for A100
    .run_commands(
        "git clone --depth 1 https://github.com/HazyResearch/ThunderKittens.git /opt/thunderkittens",
        # Configure for A100 (sm_80) and add attention kernel
        "cd /opt/thunderkittens && sed -i \"s/target = 'h100'/target = 'a100'/\" config.py",
        "cd /opt/thunderkittens && sed -i \"s/kernels = \\['fp8_gemm'\\]/kernels = ['attn_fwd', 'attn_bwd']/\" config.py || true",
        "cat /opt/thunderkittens/config.py || true",
        # Build with verbose output
        "cd /opt/thunderkittens && python setup.py install 2>&1 || echo 'TK build failed'"
    )
)

# Simpler image without TK for baseline comparison
baseline_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "numpy")
)


@app.function(image=baseline_image, gpu="A100", timeout=300)
def check_cuda_baseline():
    """Check CUDA availability on baseline image."""
    import torch

    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    # Check if flash attention is available
    try:
        from torch.nn.functional import scaled_dot_product_attention
        info["sdpa_available"] = True
    except ImportError:
        info["sdpa_available"] = False

    return info


@app.function(image=tk_image, gpu="A100", timeout=600)
def check_tk_install():
    """Check if ThunderKittens can be installed and used."""
    import subprocess
    import os

    results = {
        "cuda_check": None,
        "gcc_version": None,
        "tk_clone": None,
        "tk_build": None,
        "tk_import": None,
    }

    # Check CUDA
    try:
        import torch
        results["cuda_check"] = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    except Exception as e:
        results["cuda_check"] = f"Error: {e}"

    # Check GCC version
    try:
        gcc_out = subprocess.run(["gcc", "--version"], capture_output=True, text=True)
        results["gcc_version"] = gcc_out.stdout.split("\n")[0]
    except Exception as e:
        results["gcc_version"] = f"Error: {e}"

    # Try to clone ThunderKittens
    try:
        if not os.path.exists("/tmp/ThunderKittens"):
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/HazyResearch/ThunderKittens.git",
                 "/tmp/ThunderKittens"],
                check=True, capture_output=True
            )
        results["tk_clone"] = "Success"
    except Exception as e:
        results["tk_clone"] = f"Error: {e}"
        return results

    # Check TK structure
    tk_files = os.listdir("/tmp/ThunderKittens")
    results["tk_files"] = tk_files[:10]  # First 10 files

    return results


@app.function(image=baseline_image, gpu="A100", timeout=600)
def benchmark_attention_baseline(
    batch_size: int = 4,
    seq_len: int = 1024,
    d_model: int = 512,
    n_heads: int = 8,
    iterations: int = 100,
    warmup: int = 20
):
    """Benchmark standard PyTorch attention on A100."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda"
    dtype = torch.float16  # Use FP16 for speed

    print(f"Benchmarking on {torch.cuda.get_device_name(0)}")
    print(f"Config: B={batch_size}, N={seq_len}, D={d_model}, H={n_heads}")

    # Create test tensors
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

    # Simple MHA implementation using SDPA
    class SimpleMHA(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.out = nn.Linear(d_model, d_model)

        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Use SDPA (will use FlashAttention if available)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).reshape(B, N, C)
            return self.out(out)

    model = SimpleMHA(d_model, n_heads).to(device).to(dtype)
    model.eval()

    results = {}

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()

    # Benchmark forward pass
    print(f"Benchmarking forward ({iterations} iterations)...")
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    results["forward"] = {
        "total_ms": elapsed * 1000,
        "per_iter_ms": elapsed * 1000 / iterations,
        "throughput": iterations / elapsed
    }

    # Benchmark forward + backward
    print(f"Benchmarking forward+backward ({iterations} iterations)...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters())

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        optimizer.zero_grad()
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    results["forward_backward"] = {
        "total_ms": elapsed * 1000,
        "per_iter_ms": elapsed * 1000 / iterations,
        "throughput": iterations / elapsed
    }

    # Memory stats
    results["memory"] = {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2
    }

    results["config"] = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "n_heads": n_heads,
        "iterations": iterations,
        "device": torch.cuda.get_device_name(0),
        "dtype": str(dtype)
    }

    return results


@app.function(image=baseline_image, gpu="A100", timeout=600)
def benchmark_sequence_lengths(d_model: int = 512, n_heads: int = 8):
    """Benchmark attention at different sequence lengths."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda"
    dtype = torch.float16
    batch_size = 4
    iterations = 50
    warmup = 10

    class SimpleMHA(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.out = nn.Linear(d_model, d_model)

        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).reshape(B, N, C)
            return self.out(out)

    model = SimpleMHA(d_model, n_heads).to(device).to(dtype)
    model.eval()

    seq_lengths = [256, 512, 1024, 2048, 4096]
    results = {}

    for seq_len in seq_lengths:
        print(f"Testing seq_len={seq_len}...")
        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x)
        torch.cuda.synchronize()

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        results[seq_len] = {
            "per_iter_ms": elapsed * 1000 / iterations,
            "throughput": iterations / elapsed
        }

        # Clear cache
        torch.cuda.empty_cache()

    return {
        "results": results,
        "config": {
            "batch_size": batch_size,
            "d_model": d_model,
            "n_heads": n_heads,
            "device": torch.cuda.get_device_name(0)
        }
    }


@app.function(image=tk_image, gpu="A100", timeout=900)
def benchmark_tk_attention(
    batch_size: int = 4,
    seq_len: int = 1024,
    d_model: int = 512,
    n_heads: int = 8,
    iterations: int = 100,
    warmup: int = 20
):
    """Benchmark ThunderKittens attention vs PyTorch SDPA on A100."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import subprocess
    import sys

    device = "cuda"
    dtype = torch.float16

    print(f"Benchmarking on {torch.cuda.get_device_name(0)}")
    print(f"Config: B={batch_size}, N={seq_len}, D={d_model}, H={n_heads}")

    results = {
        "tk_available": False,
        "tk_import_error": None,
        "sdpa": {},
        "tk": {},
    }

    # Try to import ThunderKittens
    try:
        # Check if TK was installed
        import thunderkittens
        results["tk_available"] = True
        print("✓ ThunderKittens module found")

        # Try to get the attention function
        try:
            from thunderkittens import attention as tk_attn
            results["tk_attention_available"] = True
            print("✓ ThunderKittens attention kernel loaded")
        except ImportError as e:
            results["tk_attention_available"] = False
            results["tk_attention_error"] = str(e)
            print(f"⚠ TK attention not available: {e}")

    except ImportError as e:
        results["tk_import_error"] = str(e)
        print(f"⚠ ThunderKittens not installed: {e}")

        # Try to build it now
        print("Attempting to build ThunderKittens...")
        try:
            result = subprocess.run(
                ["pip", "list"],
                capture_output=True, text=True
            )
            print(f"Installed packages: {result.stdout[:500]}")
        except Exception as build_err:
            print(f"Build failed: {build_err}")

    # Create test tensors (Q, K, V format for raw attention)
    head_dim = d_model // n_heads
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)

    # Benchmark SDPA
    print(f"\nBenchmarking PyTorch SDPA ({iterations} iterations)...")

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    results["sdpa"] = {
        "per_iter_ms": elapsed * 1000 / iterations,
        "throughput": iterations / elapsed
    }
    print(f"  SDPA: {results['sdpa']['per_iter_ms']:.3f} ms/iter")

    # Benchmark TK if available
    if results.get("tk_attention_available"):
        print(f"\nBenchmarking ThunderKittens ({iterations} iterations)...")

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = tk_attn(q, k, v, causal=True)
        torch.cuda.synchronize()

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            with torch.no_grad():
                _ = tk_attn(q, k, v, causal=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        results["tk"] = {
            "per_iter_ms": elapsed * 1000 / iterations,
            "throughput": iterations / elapsed
        }
        print(f"  TK: {results['tk']['per_iter_ms']:.3f} ms/iter")

        # Compute speedup
        if results["sdpa"]["per_iter_ms"] > 0:
            speedup = results["sdpa"]["per_iter_ms"] / results["tk"]["per_iter_ms"]
            results["speedup"] = speedup
            print(f"  Speedup: {speedup:.2f}x")

    results["config"] = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "device": torch.cuda.get_device_name(0),
        "dtype": str(dtype)
    }

    return results


@app.local_entrypoint()
def main():
    """Run all benchmarks and report results."""
    print("=" * 70)
    print("ThunderKittens Modal Benchmark")
    print("=" * 70)

    # Check baseline CUDA
    print("\n1. Checking baseline CUDA setup...")
    baseline_info = check_cuda_baseline.remote()
    print(f"   PyTorch: {baseline_info['torch_version']}")
    print(f"   CUDA: {baseline_info['cuda_version']}")
    print(f"   Device: {baseline_info['device_name']}")
    print(f"   SDPA: {baseline_info['sdpa_available']}")

    # Check TK install possibility
    print("\n2. Checking ThunderKittens setup...")
    tk_info = check_tk_install.remote()
    print(f"   CUDA check: {tk_info['cuda_check']}")
    print(f"   GCC: {tk_info['gcc_version']}")
    print(f"   TK clone: {tk_info['tk_clone']}")
    if 'tk_files' in tk_info:
        print(f"   TK files: {tk_info['tk_files']}")

    # Run baseline benchmarks
    print("\n3. Running baseline attention benchmark...")
    baseline_results = benchmark_attention_baseline.remote()

    print(f"\n   Forward pass:")
    print(f"     Per iteration: {baseline_results['forward']['per_iter_ms']:.2f} ms")
    print(f"     Throughput: {baseline_results['forward']['throughput']:.1f} iter/s")

    print(f"\n   Forward + Backward:")
    print(f"     Per iteration: {baseline_results['forward_backward']['per_iter_ms']:.2f} ms")
    print(f"     Throughput: {baseline_results['forward_backward']['throughput']:.1f} iter/s")

    print(f"\n   Memory:")
    print(f"     Max allocated: {baseline_results['memory']['max_allocated_mb']:.1f} MB")

    # Benchmark different sequence lengths
    print("\n4. Benchmarking sequence length scaling...")
    seq_results = benchmark_sequence_lengths.remote()

    print(f"\n   Sequence Length Scaling (forward only):")
    print(f"   {'Seq Len':<10} {'Time (ms)':<12} {'Throughput':<12}")
    print(f"   {'-'*34}")
    for seq_len, data in seq_results['results'].items():
        print(f"   {seq_len:<10} {data['per_iter_ms']:<12.2f} {data['throughput']:<12.1f}")

    # Benchmark ThunderKittens
    print("\n5. Benchmarking ThunderKittens attention...")
    tk_results = benchmark_tk_attention.remote()

    print(f"\n   TK Available: {tk_results.get('tk_available', False)}")
    if tk_results.get('tk_import_error'):
        print(f"   Import Error: {tk_results['tk_import_error']}")

    if tk_results.get('sdpa'):
        print(f"\n   Raw Attention Kernel Comparison:")
        print(f"   SDPA: {tk_results['sdpa']['per_iter_ms']:.3f} ms/iter")

    if tk_results.get('tk') and tk_results['tk']:
        print(f"   TK:   {tk_results['tk']['per_iter_ms']:.3f} ms/iter")
        if tk_results.get('speedup'):
            print(f"   Speedup: {tk_results['speedup']:.2f}x")
    else:
        print("   TK attention not available for benchmark")

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)

    return {
        "baseline_info": baseline_info,
        "tk_info": tk_info,
        "baseline_benchmark": baseline_results,
        "seq_scaling": seq_results,
        "tk_benchmark": tk_results
    }


if __name__ == "__main__":
    # For local testing
    print("Run with: modal run scripts/test_tk_modal.py")
