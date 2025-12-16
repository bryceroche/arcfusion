#!/usr/bin/env python3
"""Quick script to train Linear Attention candidates specifically."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src/arcfusion"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from cloud_train_fair import CONFIG, app, train_model, save_result_to_db
from db import ArcFusionDB
from dream_and_train import components_to_architecture_code, get_architecture_hash, notify
import json

db_path = Path(__file__).parent.parent / "arcfusion.db"
db = ArcFusionDB(str(db_path))

# Get top 3 Linear Attention candidates by predicted efficiency
candidates = db.list_dream_candidates(untrained_only=True, limit=200)
linear_candidates = [c for c in candidates if c.has_linear_attn]

# Sort by predicted efficiency (lower is better)
for c in linear_candidates:
    c._efficiency = c.predicted_ppl * (c.predicted_time / 300) ** 0.5 if c.predicted_time > 0 else c.predicted_ppl

linear_candidates.sort(key=lambda c: c._efficiency)
top_3 = linear_candidates[:3]

print("=" * 70)
print("TRAINING TOP 3 LINEAR ATTENTION CANDIDATES")
print("=" * 70)
print(f"Selected {len(top_3)} candidates by predicted efficiency:\n")
for c in top_3:
    print(f"  {c.candidate_id}: pred_ppl={c.predicted_ppl:.1f}, pred_time={c.predicted_time:.1f}s")
    print(f"    Components: {', '.join(c.get_components()[:4])}...")
print()
sys.stdout.flush()

results = []
for i, cand in enumerate(top_3):
    print(f"\n{'=' * 70}")
    print(f"TRAINING {i+1}/{len(top_3)}: {cand.candidate_id}")
    print("=" * 70)
    sys.stdout.flush()

    # Get component objects from names
    components = []
    for name in cand.get_components():
        comp = db.find_components(name_pattern=name, min_score=0.0)
        if comp:
            components.append(comp[0])

    if not components:
        print("No valid components found, skipping...")
        continue

    code, model_name = components_to_architecture_code(components, n_layers=14)
    arch_hash = get_architecture_hash(components)
    model_name = f"{model_name}_{arch_hash}"

    print(f"Model: {model_name}")
    print(f"Components: {[c.name for c in components]}")
    sys.stdout.flush()

    try:
        with app.run():
            result = train_model.remote(code, "DreamedModel", CONFIG)

        if result["success"]:
            ppl = result["perplexity"]
            time_s = result["time_seconds"]
            print(f"  PPL: {ppl:.1f}")
            print(f"  Time: {time_s:.1f}s")

            result["model_name"] = model_name
            run_id = save_result_to_db(db, result, CONFIG, None, model_code=code)
            print(f"  Saved: {run_id}")

            db.update_dream_candidate_training(
                candidate_id=cand.candidate_id,
                training_run_id=run_id,
                actual_ppl=ppl,
                actual_time=time_s
            )

            results.append({"model_name": model_name, "ppl": ppl, "time": time_s})
        else:
            print(f"  FAILED: {result.get('error')}")
    except Exception as e:
        print(f"  ERROR: {e}")

    sys.stdout.flush()

print(f"\n{'=' * 70}")
print("LINEAR ATTENTION TRAINING COMPLETE")
print("=" * 70)
if results:
    for r in results:
        efficiency = r['ppl'] * (r['time'] / 300) ** 0.5
        print(f"  {r['model_name']}: {r['ppl']:.1f} PPL, {r['time']:.1f}s, eff={efficiency:.1f}")
    notify("Linear Attention Training Done", f"Trained {len(results)} models")
else:
    print("No models trained")
    notify("Linear Attention Training Done", "No models trained")
