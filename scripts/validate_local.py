#!/usr/bin/env python3
"""Local validation before cloud deployment.

Run a quick forward/backward pass on CPU to catch errors before wasting cloud GPU time.
Catches: syntax errors, import errors, shape mismatches, missing ops.
"""

import argparse
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


def validate_model_code(model_code: str, model_name: str, verbose: bool = True) -> dict:
    """Validate model code with a quick forward/backward pass on CPU.

    Args:
        model_code: Python code string defining a model class named model_name
        model_name: Name of the class to find in the code (e.g., 'Transformer_MHA')
        verbose: Print progress

    Returns:
        dict with success, error message, and validation details
    """
    import torch
    import torch.nn as nn

    result = {
        "success": False,
        "model_name": model_name,
        "error": None,
        "checks": {
            "code_exec": False,
            "model_build": False,
            "forward_pass": False,
            "backward_pass": False,
            "param_count": 0,
        },
    }

    # Config for small validation run (matches cloud_train_fair.py structure)
    d_model = 64  # Small for fast CPU validation
    n_heads = 4
    vocab_size = 1000
    seq_len = 32
    batch_size = 2

    try:
        # Step 1: Execute model code
        if verbose:
            print(f"[1/4] Executing model code for {model_name}...")
        ns = {}
        exec(model_code, ns)

        model_class = ns.get(model_name)
        if not model_class:
            result["error"] = f"Class '{model_name}' not found in code"
            return result
        result["checks"]["code_exec"] = True

        # Step 2: Build model
        if verbose:
            print("[2/4] Building model...")

        # The model class takes kwargs: d_model, vocab_size, n_layers, n_heads
        model = model_class(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=2,  # Small for validation
            n_heads=n_heads,
        )
        result["checks"]["param_count"] = sum(p.numel() for p in model.parameters())
        result["checks"]["model_build"] = True

        # Step 3: Forward pass
        if verbose:
            print("[3/4] Running forward pass...")
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        y = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits = model(x)

        # Check output shape
        expected_shape = (batch_size, seq_len, vocab_size)
        if logits.shape != expected_shape:
            result["error"] = f"Output shape {logits.shape} != expected {expected_shape}"
            return result
        result["checks"]["forward_pass"] = True

        # Step 4: Backward pass
        if verbose:
            print("[4/4] Running backward pass...")
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()

        # Check gradients exist
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        if grad_count == 0:
            result["error"] = "No gradients computed"
            return result
        result["checks"]["backward_pass"] = True

        result["success"] = True
        if verbose:
            print(f"✓ {model_name} passed all validation checks")
            print(f"  Parameters: {result['checks']['param_count']:,}")
            print(f"  Loss: {loss.item():.4f}")

    except Exception as e:
        import traceback
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        if verbose:
            print(f"✗ {model_name} validation FAILED")
            print(f"  Error: {e}")

    return result


def validate_all_models() -> dict:
    """Validate all models from cloud_train_fair.py"""

    # Import model definitions
    try:
        from cloud_train_fair import MODELS
    except ImportError:
        try:
            from scripts.cloud_train_fair import MODELS
        except ImportError:
            print("Error: Could not import MODELS from cloud_train_fair.py")
            print("Make sure you're running from the project root directory")
            sys.exit(1)

    results = {}
    passed = 0
    failed = 0

    print("=" * 60)
    print("LOCAL MODEL VALIDATION")
    print("=" * 60)
    print()

    for name, code in MODELS.items():
        print(f"\nValidating {name}...")
        print("-" * 40)
        result = validate_model_code(code, name)
        results[name] = result

        if result["success"]:
            passed += 1
        else:
            failed += 1

    # Summary
    print()
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{len(MODELS)}")
    print(f"Failed: {failed}/{len(MODELS)}")
    print()

    if failed > 0:
        print("Failed models:")
        for name, result in results.items():
            if not result["success"]:
                print(f"  - {name}: {result['error'][:100]}...")
        print()
        print("⚠️  Fix these errors before deploying to cloud!")
    else:
        print("✓ All models ready for cloud deployment")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate models locally before cloud deployment")
    parser.add_argument("--model", "-m", help="Validate a specific model by name")
    parser.add_argument("--code", "-c", help="Validate code from a file")
    args = parser.parse_args()

    if args.code:
        # Validate code from file
        with open(args.code) as f:
            code = f.read()
        result = validate_model_code(code, Path(args.code).stem)
        sys.exit(0 if result["success"] else 1)
    elif args.model:
        # Validate specific model
        try:
            from cloud_train_fair import MODELS
        except ImportError:
            from scripts.cloud_train_fair import MODELS
        if args.model not in MODELS:
            print(f"Error: Model '{args.model}' not found")
            print(f"Available: {', '.join(MODELS.keys())}")
            sys.exit(1)
        result = validate_model_code(MODELS[args.model], args.model)
        sys.exit(0 if result["success"] else 1)
    else:
        # Validate all models
        results = validate_all_models()
        all_passed = all(r["success"] for r in results.values())
        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
