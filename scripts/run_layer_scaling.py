#!/usr/bin/env python3
"""
Layer Scaling Comparison: MHA vs DeepGQA (arcfusion-vho)

Apples-to-apples comparison of attention mechanisms at increasing depths.

Models:
- MHA-10 vs DeepGQA-10 (10 layers each)
- MHA-14 vs DeepGQA-14 (14 layers each)
- DeepGQA-18 (pushing GQA to 18 layers)

Hypothesis: GQA efficiency advantage compounds with depth.

Run with: PYTHONUNBUFFERED=1 .venv-modal/bin/python scripts/run_layer_scaling.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "arcfusion"))

from cloud_train_fair import (
    MODELS, CONFIG, app, train_model, save_result_to_db
)
from db import ArcFusionDB

# Models to compare at each depth
LAYER_SCALING_MODELS = [
    # 10-layer comparison
    "Transformer_MHA10",
    "Transformer_DeepGQA10",  # Already exists, include for completeness
    # 14-layer comparison
    "Transformer_MHA14",
    "Transformer_DeepGQA14",
    # 18-layer (GQA only - MHA would be too slow)
    "Transformer_DeepGQA18",
]

def main():
    print("=" * 70)
    print("LAYER SCALING COMPARISON: MHA vs DeepGQA")
    print("=" * 70)
    print("Testing if GQA efficiency advantage grows with depth")
    print()
    print("Comparison pairs:")
    print("  10 layers: MHA10 vs DeepGQA10")
    print("  14 layers: MHA14 vs DeepGQA14")
    print("  18 layers: DeepGQA18 (GQA only)")
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

    print(f"Baseline: MHA (4-layer) = {baseline_ppl:.1f} PPL")
    print()
    sys.stdout.flush()

    results = []

    for model_name in LAYER_SCALING_MODELS:
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
            print(f"  vs MHA baseline: {vs_baseline:+.1f}%")

            # Save to DB
            run_id = save_result_to_db(db, result, CONFIG, baseline_run_id, model_code=code)
            print(f"  Saved: {run_id}")

            # Extract layer count from model name
            if "10" in model_name:
                layers = 10
            elif "14" in model_name:
                layers = 14
            elif "18" in model_name:
                layers = 18
            else:
                layers = 0

            results.append({
                "name": model_name,
                "layers": layers,
                "attn_type": "GQA" if "GQA" in model_name else "MHA",
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
    print("LAYER SCALING SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Layers':>6} {'Type':>5} {'PPL':>8} {'Time':>8} {'vs MHA':>10}")
    print("-" * 70)

    # Sort by layers then by attention type
    for r in sorted(results, key=lambda x: (x["layers"], x["attn_type"])):
        print(f"{r['name']:<25} {r['layers']:>6} {r['attn_type']:>5} {r['ppl']:>8.1f} {r['time']:>7.0f}s {r['vs_baseline']:>+9.1f}%")

    print("-" * 70)
    print(f"{'MHA baseline (4-layer)':<25} {4:>6} {'MHA':>5} {baseline_ppl:>8.1f} {197:>7.0f}s {'0.0%':>10}")
    print()

    # Analysis by layer count
    print("=" * 70)
    print("ANALYSIS BY DEPTH")
    print("=" * 70)

    for layer_count in [10, 14]:
        mha = next((r for r in results if r["layers"] == layer_count and r["attn_type"] == "MHA"), None)
        gqa = next((r for r in results if r["layers"] == layer_count and r["attn_type"] == "GQA"), None)

        if mha and gqa:
            ppl_diff = ((gqa["ppl"] - mha["ppl"]) / mha["ppl"]) * 100
            time_diff = ((gqa["time"] - mha["time"]) / mha["time"]) * 100
            efficiency = (mha["ppl"] / mha["time"]) / (gqa["ppl"] / gqa["time"])

            print(f"\n{layer_count} layers:")
            print(f"  MHA{layer_count}: {mha['ppl']:.1f} PPL @ {mha['time']:.0f}s")
            print(f"  GQA{layer_count}: {gqa['ppl']:.1f} PPL @ {gqa['time']:.0f}s")
            print(f"  GQA vs MHA: {ppl_diff:+.1f}% PPL, {time_diff:+.1f}% time")
            print(f"  GQA efficiency multiplier: {efficiency:.2f}x")

    # DeepGQA18 standalone
    gqa18 = next((r for r in results if r["layers"] == 18), None)
    if gqa18:
        print(f"\n18 layers (GQA only):")
        print(f"  DeepGQA18: {gqa18['ppl']:.1f} PPL @ {gqa18['time']:.0f}s")
        print(f"  vs baseline: {gqa18['vs_baseline']:+.1f}%")

    # Final verdict
    print("\n" + "=" * 70)
    gqa_results = [r for r in results if r["attn_type"] == "GQA"]
    mha_results = [r for r in results if r["attn_type"] == "MHA"]

    if gqa_results and mha_results:
        avg_gqa_ppl = sum(r["ppl"] for r in gqa_results) / len(gqa_results)
        avg_mha_ppl = sum(r["ppl"] for r in mha_results) / len(mha_results)
        avg_gqa_time = sum(r["time"] for r in gqa_results) / len(gqa_results)
        avg_mha_time = sum(r["time"] for r in mha_results) / len(mha_results)

        if avg_gqa_ppl < avg_mha_ppl and avg_gqa_time < avg_mha_time:
            print("VERDICT: GQA dominates - better quality AND faster!")
        elif avg_gqa_ppl < avg_mha_ppl:
            print("VERDICT: GQA wins on quality")
        elif avg_gqa_time < avg_mha_time:
            print("VERDICT: GQA wins on speed")
        else:
            print("VERDICT: Results inconclusive")

    sys.stdout.flush()

if __name__ == "__main__":
    main()
