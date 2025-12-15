#!/usr/bin/env python3
"""Batch training for dreamed architectures.

Run overnight to train multiple architectures and compare results.

Usage:
    # Train from a JSON file with model definitions
    python scripts/batch_train.py --input experiments/dreamed_models.json

    # Train a single model from a Python file
    python scripts/batch_train.py --code model.py --name MyModel

    # List models in the built-in set
    python scripts/batch_train.py --list

    # Train built-in models (subset)
    python scripts/batch_train.py --builtin MHA,GQA

Example JSON input format:
{
    "models": {
        "MyModel_v1": "import torch\\nimport torch.nn as nn\\n...",
        "MyModel_v2": "import torch\\nimport torch.nn as nn\\n..."
    },
    "config": {  // optional overrides
        "max_steps": 1000,
        "d_model": 128
    }
}
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


def validate_locally(models: dict[str, str], verbose: bool = True) -> dict:
    """Validate all models locally before cloud training.

    Returns dict mapping model_name -> validation_result
    """
    try:
        from validate_local import validate_model_code
    except ImportError:
        from scripts.validate_local import validate_model_code

    results = {}
    passed = 0
    failed = 0

    if verbose:
        print("\n" + "=" * 60)
        print("LOCAL VALIDATION (catching errors before cloud spend)")
        print("=" * 60)

    for name, code in models.items():
        if verbose:
            print(f"\nValidating {name}...")
        result = validate_model_code(code, name, verbose=verbose)
        results[name] = result

        if result["success"]:
            passed += 1
        else:
            failed += 1

    if verbose:
        print(f"\n{'-' * 60}")
        print(f"Validation: {passed} passed, {failed} failed")
        if failed > 0:
            print("\nFailed models (will be skipped):")
            for name, r in results.items():
                if not r["success"]:
                    err = r["error"] or "Unknown error"
                    print(f"  - {name}: {err[:80]}...")

    return results


def train_batch(
    models: dict[str, str],
    config_overrides: dict | None = None,
    skip_validation: bool = False,
    output_path: str = "experiments/batch_results.json",
) -> dict:
    """Train a batch of models on Modal cloud.

    Args:
        models: dict mapping model_name -> model_code
        config_overrides: optional config overrides (max_steps, d_model, etc.)
        skip_validation: skip local validation (not recommended)
        output_path: where to save results

    Returns:
        dict with training results for each model
    """
    import modal

    # Import training infrastructure
    try:
        from cloud_train_fair import (
            app, train_model, CONFIG, BASELINE_MODEL,
            config_hash, save_result_to_db
        )
    except ImportError:
        from scripts.cloud_train_fair import (
            app, train_model, CONFIG, BASELINE_MODEL,
            config_hash, save_result_to_db
        )

    # Import DB
    sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "arcfusion"))
    from db import ArcFusionDB

    # Apply config overrides
    config = CONFIG.copy()
    if config_overrides:
        config.update(config_overrides)

    print("\n" + "=" * 70)
    print("BATCH TRAINING")
    print("=" * 70)
    print(f"Models to train: {len(models)}")
    print(f"Config: {config['n_layers']} layers, d_model={config['d_model']}, {config['max_steps']} steps")
    print(f"GPU: {config.get('gpu', 'A100')}, Mixed Precision: {config.get('mixed_precision', True)}")

    # Step 1: Local validation
    if not skip_validation:
        validation_results = validate_locally(models, verbose=True)
        models_to_train = {
            name: code
            for name, code in models.items()
            if validation_results.get(name, {}).get("success", False)
        }
        skipped = len(models) - len(models_to_train)
        if skipped > 0:
            print(f"\nSkipping {skipped} models that failed validation")
    else:
        print("\nWARNING: Skipping local validation (not recommended)")
        models_to_train = models

    if not models_to_train:
        print("\nNo models to train!")
        return {"results": {}, "config": config, "error": "All models failed validation"}

    # Step 2: Get baseline stats from DB
    db = ArcFusionDB("arcfusion.db")
    cfg_hash = config_hash(config)
    baseline_stats = db.get_baseline_stats(BASELINE_MODEL, cfg_hash)

    print(f"\n{'-' * 70}")
    print(f"Baseline: {BASELINE_MODEL}")
    if baseline_stats['n_runs'] > 0:
        print(f"  Mean perplexity: {baseline_stats['mean_ppl']:.2f} ± {baseline_stats['std_ppl']:.2f}")
    else:
        print("  No baseline data - will need to train baseline first")

    # Step 3: Train each model
    results = {}
    start_time = datetime.now()

    for i, (name, code) in enumerate(models_to_train.items(), 1):
        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(models_to_train)}] Training: {name}")
        print("=" * 70)

        try:
            with app.run():
                result = train_model.remote(code, name, config)

            results[name] = result

            if result["success"]:
                ppl = result["perplexity"]
                time_s = result["time_seconds"]
                params = result["parameters"]

                print(f"  Perplexity: {ppl:.2f}")
                print(f"  Time: {time_s:.1f}s")
                print(f"  Parameters: {params:,}")

                # Compare to baseline
                if baseline_stats['mean_ppl'] > 0:
                    delta = ppl - baseline_stats['mean_ppl']
                    pct = (delta / baseline_stats['mean_ppl']) * 100
                    print(f"  vs Baseline: {pct:+.1f}%")

                # Save to DB
                run_id = save_result_to_db(db, result, config)
                print(f"  Saved: {run_id}")
            else:
                print(f"  FAILED: {result.get('error', 'Unknown error')[:200]}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {
                "success": False,
                "model_name": name,
                "error": str(e),
            }

        # Save intermediate results after each model
        _save_results(results, config, baseline_stats, output_path)

    elapsed = (datetime.now() - start_time).total_seconds()

    # Step 4: Summary
    print("\n" + "=" * 70)
    print("BATCH TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Models trained: {sum(1 for r in results.values() if r.get('success', False))}/{len(models_to_train)}")

    # Rank by perplexity
    successful = [(n, r) for n, r in results.items() if r.get("success")]
    if successful:
        ranked = sorted(successful, key=lambda x: x[1].get("perplexity", float("inf")))
        print(f"\nRankings (by perplexity):")
        for i, (name, r) in enumerate(ranked, 1):
            ppl = r["perplexity"]
            if baseline_stats['mean_ppl'] > 0:
                pct = ((ppl - baseline_stats['mean_ppl']) / baseline_stats['mean_ppl']) * 100
                print(f"  {i}. {name}: {ppl:.2f} ({pct:+.1f}% vs baseline)")
            else:
                print(f"  {i}. {name}: {ppl:.2f}")

    # Final save
    _save_results(results, config, baseline_stats, output_path)
    print(f"\nResults saved to: {output_path}")

    db.close()
    return {"results": results, "config": config, "baseline": baseline_stats}


def _save_results(results: dict, config: dict, baseline_stats: dict, output_path: str):
    """Save results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "baseline": {
            "model": "Transformer_MHA",
            "mean_ppl": baseline_stats.get("mean_ppl"),
            "std_ppl": baseline_stats.get("std_ppl"),
            "n_runs": baseline_stats.get("n_runs", 0),
        },
        "results": results,
    }
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)


def load_models_from_json(path: str) -> tuple[dict[str, str], dict | None]:
    """Load models from a JSON file.

    Returns:
        (models_dict, config_overrides)
    """
    with open(path) as f:
        data = json.load(f)

    if "models" in data:
        models = data["models"]
        config = data.get("config")
    else:
        # Assume it's just a flat dict of models
        models = data
        config = None

    return models, config


def load_model_from_file(path: str, name: str | None = None) -> dict[str, str]:
    """Load a single model from a Python file."""
    with open(path) as f:
        code = f.read()

    model_name = name or Path(path).stem
    return {model_name: code}


def main():
    parser = argparse.ArgumentParser(
        description="Batch train architectures on Modal cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", help="JSON file with model definitions")
    parser.add_argument("--code", "-c", help="Single Python file with model code")
    parser.add_argument("--name", "-n", help="Model name (with --code)")
    parser.add_argument("--builtin", "-b", help="Comma-separated list of built-in models (MHA,GQA,MQA,Mamba,Hybrid)")
    parser.add_argument("--list", "-l", action="store_true", help="List built-in models")
    parser.add_argument("--output", "-o", default="experiments/batch_results.json", help="Output JSON path")
    parser.add_argument("--skip-validation", action="store_true", help="Skip local validation")
    parser.add_argument("--max-steps", type=int, help="Override max training steps")
    parser.add_argument("--d-model", type=int, help="Override model dimension")
    args = parser.parse_args()

    # Import built-in models
    try:
        from cloud_train_fair import MODELS
    except ImportError:
        from scripts.cloud_train_fair import MODELS

    # Handle --list
    if args.list:
        print("Built-in models:")
        for name in MODELS.keys():
            short_name = name.replace("Transformer_", "")
            print(f"  {short_name} ({name})")
        return

    # Determine what to train
    models = {}
    config_overrides = {}

    if args.input:
        models, file_config = load_models_from_json(args.input)
        if file_config:
            config_overrides.update(file_config)

    elif args.code:
        models = load_model_from_file(args.code, args.name)

    elif args.builtin:
        names = [n.strip() for n in args.builtin.split(",")]
        for name in names:
            full_name = f"Transformer_{name}" if not name.startswith("Transformer_") else name
            if full_name in MODELS:
                models[full_name] = MODELS[full_name]
            else:
                print(f"Warning: Unknown built-in model '{name}'")

    else:
        parser.print_help()
        print("\nError: Must specify --input, --code, or --builtin")
        sys.exit(1)

    # Apply CLI config overrides
    if args.max_steps:
        config_overrides["max_steps"] = args.max_steps
    if args.d_model:
        config_overrides["d_model"] = args.d_model

    # Run batch training
    train_batch(
        models=models,
        config_overrides=config_overrides if config_overrides else None,
        skip_validation=args.skip_validation,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
