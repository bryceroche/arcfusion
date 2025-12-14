#!/usr/bin/env python3
"""Test WikiText-2 caching on Modal."""

import modal

modal.enable_output()

app = modal.App("arcfusion-cache-test-v3")

# Pre-cache WikiText-2 during image build
# Use default HF cache location so it persists to runtime
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "numpy", "datasets", "tiktoken")
    .run_commands(
        # Download to default cache location (~/.cache/huggingface)
        'python -c "from datasets import load_dataset; load_dataset(\'wikitext\', \'wikitext-2-raw-v1\', trust_remote_code=True)"',
    )
)


@app.function(image=image, timeout=120)
def test_cached_data():
    """Test that data is pre-cached and loads instantly."""
    import time
    from datasets import load_dataset

    start = time.time()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    elapsed = time.time() - start

    return {
        "load_time_seconds": elapsed,
        "train_samples": len(ds["train"]),
        "cached": elapsed < 5.0,  # Should be instant if cached
    }


def main():
    print("Testing cached data loading...")
    with app.run():
        result = test_cached_data.remote()

    print(f"Load time: {result['load_time_seconds']:.2f}s")
    print(f"Train samples: {result['train_samples']}")
    print(f"Was cached: {result['cached']}")


if __name__ == "__main__":
    main()
