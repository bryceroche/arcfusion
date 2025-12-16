#!/usr/bin/env python3
"""Quick run of MHA18 for 18-layer comparison with DeepGQA18"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "arcfusion"))

from cloud_train_fair import MODELS, CONFIG, app, train_model, save_result_to_db
from db import ArcFusionDB

def main():
    print("=" * 70)
    print("18-LAYER COMPARISON: MHA18 vs DeepGQA18")
    print("=" * 70)
    print("DeepGQA18: 231.9 PPL @ 129s (already ran)")
    print("Running MHA18...")
    sys.stdout.flush()

    db_path = Path(__file__).parent.parent / "arcfusion.db"
    db = ArcFusionDB(str(db_path))

    baseline_runs = db.list_training_runs(model_name="Transformer_MHA", success_only=True, limit=1)
    baseline_run_id = baseline_runs[0].run_id if baseline_runs else ""
    baseline_ppl = baseline_runs[0].perplexity if baseline_runs else 274.8

    code = MODELS["Transformer_MHA18"]

    with app.run():
        result = train_model.remote(code, "Transformer_MHA18", CONFIG)

    if result["success"]:
        ppl = result["perplexity"]
        time_s = result["time_seconds"]
        vs_baseline = ((ppl - baseline_ppl) / baseline_ppl) * 100
        print(f"\nMHA18: {ppl:.1f} PPL @ {time_s:.1f}s ({vs_baseline:+.1f}% vs baseline)")

        run_id = save_result_to_db(db, result, CONFIG, baseline_run_id, model_code=code)
        print(f"Saved: {run_id}")

        # Compare with DeepGQA18
        gqa18_ppl = 231.9
        gqa18_time = 129.4
        ppl_diff = ((ppl - gqa18_ppl) / gqa18_ppl) * 100
        time_diff = ((time_s - gqa18_time) / gqa18_time) * 100

        print(f"\n18-LAYER COMPARISON:")
        print(f"  MHA18:     {ppl:.1f} PPL @ {time_s:.0f}s")
        print(f"  DeepGQA18: {gqa18_ppl:.1f} PPL @ {gqa18_time:.0f}s")
        print(f"  MHA vs GQA: {ppl_diff:+.1f}% PPL, {time_diff:+.1f}% time")
    else:
        print(f"FAILED: {result['error']}")

if __name__ == "__main__":
    main()
