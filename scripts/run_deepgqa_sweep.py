#!/usr/bin/env python3
"""
DeepGQA Depth Sweep: Test if DeepGQA6 quality scales with more layers.

DeepGQA6 breakthrough: 270.3 PPL @ 62s (beats MHA baseline 274.8)
- DeepGQA8: 8 GQA layers, expected ~112s
- DeepGQA10: 10 GQA layers, expected ~140s

Run with: PYTHONUNBUFFERED=1 .venv-modal/bin/python scripts/run_deepgqa_sweep.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "arcfusion"))

from cloud_train_fair import (
    MODELS, CONFIG, app, train_model, save_result_to_db
)
from db import ArcFusionDB

# Models to test in depth sweep
DEPTH_SWEEP_MODELS = [
    "Transformer_DeepGQA8",
    "Transformer_DeepGQA10",
]

def main():
    print("=" * 70)
    print("DeepGQA DEPTH SWEEP")
    print("=" * 70)
    print("Testing if DeepGQA6 quality scales with more depth")
    print(f"Reference: DeepGQA6 = 270.3 PPL @ 62s (beats MHA 274.8)")
    print("=" * 70)
    print()
    sys.stdout.flush()

    # Connect to DB
    db_path = Path(__file__).parent.parent / "arcfusion.db"
    db = ArcFusionDB(str(db_path))

    # Get baseline for comparison
    baseline_runs = db.list_training_runs(model_name="Transformer_MHA", success_only=True, limit=1)
    baseline_run_id = baseline_runs[0].run_id if baseline_runs else ""
    baseline_ppl = baseline_runs[0].perplexity if baseline_runs else 274.8

    print(f"Baseline: MHA = {baseline_ppl:.1f} PPL")
    print()
    sys.stdout.flush()

    results = []

    for model_name in DEPTH_SWEEP_MODELS:
        if model_name not in MODELS:
            print(f"SKIP: {model_name} not found in MODELS dict")
            sys.stdout.flush()
            continue

        code = MODELS[model_name]
        print(f"Training {model_name}...")
        print("-" * 50)
        sys.stdout.flush()

        # Run Modal function within app context
        with app.run():
            result = train_model.remote(code, model_name, CONFIG)

        if result["success"]:
            ppl = result["perplexity"]
            time_s = result["time_seconds"]
            vs_baseline = ((ppl - baseline_ppl) / baseline_ppl) * 100

            print(f"  PPL: {ppl:.1f}")
            print(f"  Time: {time_s:.1f}s")
            print(f"  vs MHA: {vs_baseline:+.1f}%")

            # Save to DB
            run_id = save_result_to_db(db, result, CONFIG, baseline_run_id, model_code=code)
            print(f"  Saved: {run_id}")

            results.append({
                "name": model_name,
                "ppl": ppl,
                "time": time_s,
                "vs_baseline": vs_baseline,
            })
        else:
            print(f"  FAILED: {result['error']}")

        print()
        sys.stdout.flush()

    # Summary
    print("=" * 70)
    print("DEPTH SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'PPL':>8} {'Time':>8} {'vs MHA':>10}")
    print("-" * 55)
    print(f"{'DeepGQA6 (reference)':25} {270.3:>8.1f} {62:>7.0f}s {-1.6:>+9.1f}%")
    for r in results:
        print(f"{r['name']:25} {r['ppl']:>8.1f} {r['time']:>7.0f}s {r['vs_baseline']:>+9.1f}%")
    print("-" * 55)
    print(f"{'MHA baseline':25} {baseline_ppl:>8.1f} {197:>7.0f}s {'0.0%':>10}")
    print()

    # Analysis
    if results:
        best = min(results, key=lambda x: x["ppl"])
        print(f"Best in sweep: {best['name']} ({best['ppl']:.1f} PPL)")

        if best["ppl"] < 270.3:
            print("BREAKTHROUGH: Deeper GQA beats DeepGQA6!")
        elif best["ppl"] < baseline_ppl:
            print("Still beats MHA baseline, but DeepGQA6 remains optimal.")
        else:
            print("Quality degraded with more depth - diminishing returns.")

    sys.stdout.flush()

if __name__ == "__main__":
    main()
