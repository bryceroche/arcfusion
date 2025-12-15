#!/usr/bin/env python3
"""Score and rank benchmark results combining perplexity and training time."""

import json
import math
from pathlib import Path


def load_results(path: str = "experiments/fair_comparison_results.json") -> dict:
    """Load benchmark results."""
    with open(path) as f:
        return json.load(f)


def calculate_scores(results: dict) -> list[dict]:
    """Calculate combined scores for each model.

    Scoring approaches:
    1. efficiency_score: How much perplexity improvement per unit time
       = (baseline_ppl - model_ppl) / sqrt(time_seconds)
       Higher is better (more improvement, less time)

    2. normalized_score: Weighted combination of normalized metrics
       = 0.7 * ppl_improvement_pct + 0.3 * time_efficiency_pct
       Higher is better
    """
    baseline = results["baseline_stats"]
    baseline_ppl = baseline["mean_ppl"]

    # Get baseline time (assume ~60s for MHA)
    baseline_time = 60.0
    if "Transformer_MHA" in results["results"]:
        mha = results["results"]["Transformer_MHA"]
        baseline_time = mha.get("time_seconds", 60.0)

    scored = []

    for name, data in results["results"].items():
        if not data.get("success", False):
            continue

        ppl = data.get("perplexity", float("inf"))
        time_s = data.get("time_seconds", baseline_time)
        params = data.get("parameters", 0)

        # Perplexity improvement (positive = better than baseline)
        ppl_improvement = baseline_ppl - ppl
        ppl_improvement_pct = (ppl_improvement / baseline_ppl) * 100

        # Time ratio (>1 means slower than baseline)
        time_ratio = time_s / baseline_time if baseline_time > 0 else 1.0

        # Efficiency score: improvement per sqrt(time)
        # Using sqrt to not penalize slow models too harshly
        efficiency = ppl_improvement / math.sqrt(time_s) if time_s > 0 else 0

        # Normalized score (0-100 scale, higher = better)
        # PPL component: -20% improvement = +20 points, +5% = -5 points
        ppl_score = ppl_improvement_pct * 1.0  # 1:1 mapping

        # Time component: Same time = 0, 10x slower = -10 points
        time_score = -10 * math.log10(time_ratio) if time_ratio > 0 else 0

        # Combined: weight perplexity more (70/30)
        combined_score = 0.7 * ppl_score + 0.3 * time_score

        scored.append({
            "name": name,
            "perplexity": ppl,
            "ppl_vs_baseline": f"{ppl_improvement_pct:+.1f}%",
            "time_seconds": time_s,
            "time_ratio": f"{time_ratio:.1f}x",
            "efficiency": efficiency,
            "combined_score": combined_score,
            "parameters": params,
            "is_baseline": data.get("is_baseline", False),
        })

    # Sort by combined score (higher = better)
    scored.sort(key=lambda x: x["combined_score"], reverse=True)

    return scored


def print_ranked_table(scored: list[dict]) -> None:
    """Print a nice ranked table."""
    print("\n" + "=" * 85)
    print("BENCHMARK RANKINGS (Combined Score = 0.7 * PPL improvement + 0.3 * Time efficiency)")
    print("=" * 85)
    print(f"{'Rank':<5} {'Model':<20} {'PPL':<10} {'vs Base':<10} {'Time':<10} {'Score':<10}")
    print("-" * 85)

    for i, r in enumerate(scored, 1):
        marker = " *" if r["is_baseline"] else ""
        print(f"{i:<5} {r['name']:<20} {r['perplexity']:<10.2f} {r['ppl_vs_baseline']:<10} "
              f"{r['time_ratio']:<10} {r['combined_score']:<+10.2f}{marker}")

    print("-" * 85)
    print("* = baseline model")
    print("\nScore interpretation:")
    print("  > 0: Better than baseline overall")
    print("  = 0: Same as baseline")
    print("  < 0: Worse than baseline overall")


def print_efficiency_table(scored: list[dict]) -> None:
    """Print efficiency-focused table (improvement per sqrt(time))."""
    print("\n" + "=" * 75)
    print("EFFICIENCY RANKINGS (PPL improvement / sqrt(time))")
    print("=" * 75)

    # Sort by efficiency
    by_efficiency = sorted(scored, key=lambda x: x["efficiency"], reverse=True)

    print(f"{'Rank':<5} {'Model':<20} {'PPL Δ':<12} {'Time':<12} {'Efficiency':<12}")
    print("-" * 75)

    for i, r in enumerate(by_efficiency, 1):
        ppl_delta = float(r["ppl_vs_baseline"].rstrip("%"))
        print(f"{i:<5} {r['name']:<20} {ppl_delta:>+7.1f}%     {r['time_seconds']:>6.0f}s     "
              f"{r['efficiency']:>+8.3f}")

    print("-" * 75)


def main():
    results = load_results()
    scored = calculate_scores(results)

    print_ranked_table(scored)
    print_efficiency_table(scored)

    # Save scored results
    output_path = Path("experiments/scored_results.json")
    with open(output_path, "w") as f:
        json.dump(scored, f, indent=2)
    print(f"\nScored results saved to {output_path}")


if __name__ == "__main__":
    main()
