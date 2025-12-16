#!/usr/bin/env python3
"""
12-Layer comparison: MHA12 vs DeepGQA12

Run with: PYTHONUNBUFFERED=1 .venv-modal/bin/python scripts/run_12layer.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "arcfusion"))

from cloud_train_fair import (
    MODELS, CONFIG, app, train_model, save_result_to_db
)
from db import ArcFusionDB

MODELS_12 = ["Transformer_MHA12", "Transformer_DeepGQA12"]

def main():
    print("=" * 70)
    print("12-LAYER COMPARISON: MHA12 vs DeepGQA12")
    print("=" * 70)
    sys.stdout.flush()

    db_path = Path(__file__).parent.parent / "arcfusion.db"
    db = ArcFusionDB(str(db_path))

    baseline_runs = db.list_training_runs(model_name="Transformer_MHA", success_only=True, limit=1)
    baseline_run_id = baseline_runs[0].run_id if baseline_runs else ""
    baseline_ppl = baseline_runs[0].perplexity if baseline_runs else 274.8

    print(f"Baseline: MHA (4-layer) = {baseline_ppl:.1f} PPL\n")
    sys.stdout.flush()

    results = []
    for model_name in MODELS_12:
        if model_name not in MODELS:
            print(f"SKIP: {model_name} not found")
            continue

        code = MODELS[model_name]
        print(f"Training {model_name}...")
        print("-" * 50)
        sys.stdout.flush()

        with app.run():
            result = train_model.remote(code, model_name, CONFIG)

        if result["success"]:
            ppl = result["perplexity"]
            time_s = result["time_seconds"]
            vs_baseline = ((ppl - baseline_ppl) / baseline_ppl) * 100
            print(f"  PPL: {ppl:.1f}")
            print(f"  Time: {time_s:.1f}s")
            print(f"  vs MHA baseline: {vs_baseline:+.1f}%")
            run_id = save_result_to_db(db, result, CONFIG, baseline_run_id, model_code=code)
            print(f"  Saved: {run_id}")
            results.append({"name": model_name, "ppl": ppl, "time": time_s})
        else:
            print(f"  FAILED: {result['error']}")
        print()
        sys.stdout.flush()

    # Summary
    if len(results) == 2:
        mha = results[0]
        gqa = results[1]
        print("=" * 70)
        print("12-LAYER SUMMARY")
        print("=" * 70)
        print(f"MHA12:     {mha['ppl']:.1f} PPL @ {mha['time']:.0f}s")
        print(f"DeepGQA12: {gqa['ppl']:.1f} PPL @ {gqa['time']:.0f}s")
        ppl_diff = ((gqa['ppl'] - mha['ppl']) / mha['ppl']) * 100
        time_diff = ((gqa['time'] - mha['time']) / mha['time']) * 100
        print(f"GQA vs MHA: {ppl_diff:+.1f}% PPL, {time_diff:+.1f}% time")

if __name__ == "__main__":
    main()
