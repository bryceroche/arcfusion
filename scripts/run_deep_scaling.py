#!/usr/bin/env python3
"""
Deep Layer Scaling: MHA vs GQA at 24, 32, 40, 48 layers

Testing if quality gap or speed gap changes at extreme depths.

Run with: PYTHONUNBUFFERED=1 .venv-modal/bin/python scripts/run_deep_scaling.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "arcfusion"))

from cloud_train_fair import (
    MODELS, CONFIG, app, train_model, save_result_to_db
)
from db import ArcFusionDB

# Deep scaling models: 24, 32, 40, 48 layers
DEEP_SCALING_MODELS = [
    ("Transformer_MHA24", "Transformer_DeepGQA24", 24),
    ("Transformer_MHA32", "Transformer_DeepGQA32", 32),
    ("Transformer_MHA40", "Transformer_DeepGQA40", 40),
    ("Transformer_MHA48", "Transformer_DeepGQA48", 48),
]

def main():
    print("=" * 70)
    print("DEEP LAYER SCALING: MHA vs GQA (24, 32, 40, 48 layers)")
    print("=" * 70)
    print("Testing if quality/speed gap changes at extreme depths")
    print()
    sys.stdout.flush()

    # Connect to DB
    db_path = Path(__file__).parent.parent / "arcfusion.db"
    db = ArcFusionDB(str(db_path))

    # Get baseline for comparison
    baseline_runs = db.list_training_runs(model_name="Transformer_MHA", success_only=True, limit=1)
    baseline_run_id = baseline_runs[0].run_id if baseline_runs else ""
    baseline_ppl = baseline_runs[0].perplexity if baseline_runs else 274.8

    print(f"Baseline: MHA (4-layer) = {baseline_ppl:.1f} PPL")
    print()
    sys.stdout.flush()

    results = []

    for mha_name, gqa_name, layers in DEEP_SCALING_MODELS:
        print(f"\n{'=' * 70}")
        print(f"{layers}-LAYER COMPARISON")
        print("=" * 70)
        sys.stdout.flush()

        pair_results = {}

        for model_name in [mha_name, gqa_name]:
            if model_name not in MODELS:
                print(f"SKIP: {model_name} not found")
                continue

            code = MODELS[model_name]
            attn_type = "GQA" if "GQA" in model_name else "MHA"
            print(f"\nTraining {model_name}...")
            sys.stdout.flush()

            with app.run():
                result = train_model.remote(code, model_name, CONFIG)

            if result["success"]:
                ppl = result["perplexity"]
                time_s = result["time_seconds"]
                vs_baseline = ((ppl - baseline_ppl) / baseline_ppl) * 100

                print(f"  PPL: {ppl:.1f}")
                print(f"  Time: {time_s:.1f}s")
                print(f"  vs baseline: {vs_baseline:+.1f}%")

                run_id = save_result_to_db(db, result, CONFIG, baseline_run_id, model_code=code)
                print(f"  Saved: {run_id}")

                pair_results[attn_type] = {"ppl": ppl, "time": time_s}
                results.append({
                    "name": model_name,
                    "layers": layers,
                    "attn_type": attn_type,
                    "ppl": ppl,
                    "time": time_s,
                    "vs_baseline": vs_baseline,
                })
            else:
                print(f"  FAILED: {result['error']}")

            sys.stdout.flush()

        # Compare the pair
        if "MHA" in pair_results and "GQA" in pair_results:
            mha = pair_results["MHA"]
            gqa = pair_results["GQA"]
            ppl_diff = ((gqa["ppl"] - mha["ppl"]) / mha["ppl"]) * 100
            time_diff = ((gqa["time"] - mha["time"]) / mha["time"]) * 100
            print(f"\n{layers}L Summary: GQA vs MHA = {ppl_diff:+.1f}% PPL, {time_diff:+.1f}% time")

    # Final summary
    print("\n" + "=" * 70)
    print("DEEP SCALING SUMMARY")
    print("=" * 70)
    print(f"{'Layers':>6} | {'MHA PPL':>10} {'MHA Time':>10} | {'GQA PPL':>10} {'GQA Time':>10} | {'PPL Diff':>10} {'Time Diff':>10}")
    print("-" * 80)

    for layers in [24, 32, 40, 48]:
        mha = next((r for r in results if r["layers"] == layers and r["attn_type"] == "MHA"), None)
        gqa = next((r for r in results if r["layers"] == layers and r["attn_type"] == "GQA"), None)

        if mha and gqa:
            ppl_diff = ((gqa["ppl"] - mha["ppl"]) / mha["ppl"]) * 100
            time_diff = ((gqa["time"] - mha["time"]) / mha["time"]) * 100
            print(f"{layers:>6} | {mha['ppl']:>10.1f} {mha['time']:>9.0f}s | {gqa['ppl']:>10.1f} {gqa['time']:>9.0f}s | {ppl_diff:>+9.1f}% {time_diff:>+9.1f}%")
        elif mha:
            print(f"{layers:>6} | {mha['ppl']:>10.1f} {mha['time']:>9.0f}s | {'---':>10} {'---':>10} | {'---':>10} {'---':>10}")
        elif gqa:
            print(f"{layers:>6} | {'---':>10} {'---':>10} | {gqa['ppl']:>10.1f} {gqa['time']:>9.0f}s | {'---':>10} {'---':>10}")

    print()
    sys.stdout.flush()

if __name__ == "__main__":
    main()
