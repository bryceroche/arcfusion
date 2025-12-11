# ArcFusion

**ML Architecture Component Database & Composer**

Decompose ML architectures into reusable components, track relationships, dream up new configurations, and generate PyTorch code.

[![Tests](https://img.shields.io/badge/tests-55%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## The Idea

Think of the Transformer as a Formula One engine. It's made of distinct components:
- **Embedding** - Token + positional embeddings
- **MultiHeadAttention** - The crown jewel
- **FeedForward** - Position-wise MLP
- **LayerNorm** - Stabilizes training
- **ResidualConnection** - Skip connections

ArcFusion lets you:
1. **Decompose** - Break architectures into components (manually or via LLM)
2. **Store** - Track components and their relationships in SQLite
3. **Compose** - Dream up new architectures by combining components
4. **Generate** - Output PyTorch code from dreamed architectures
5. **Deduplicate** - Fuzzy matching to clean up duplicate components
6. **Benchmark** - Track what works and what doesn't

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Initialize database with Transformer, Mamba, RWKV, LLaMA, etc.
arcfusion init

# Show stats
arcfusion stats

# List components
arcfusion list components

# List engines (architectures)
arcfusion list engines

# Show details
arcfusion show Transformer
arcfusion show MultiHeadAttention

# Dream up new architectures
arcfusion dream greedy --start Attention
arcfusion dream random --steps 5 --temperature 0.8
arcfusion dream mutate --engine Transformer --rate 0.3
arcfusion dream crossover --engine1 BERT --engine2 Mamba

# Generate PyTorch code from dream
arcfusion generate greedy -o model.py
arcfusion generate crossover --engine1 BERT --engine2 Mamba -n HybridModel -o hybrid.py

# LLM-powered analysis (requires ANTHROPIC_API_KEY)
arcfusion analyze --ids 1706.03762  # Transformer paper

# Find and merge duplicates
arcfusion dedup              # Preview
arcfusion dedup --apply      # Apply changes
```

## Python API

```python
from arcfusion import ArcFusionDB, EngineComposer, seed_transformers, seed_modern_architectures

# Initialize
db = ArcFusionDB("arcfusion.db")
seed_transformers(db)
seed_modern_architectures(db)

# Query
components = db.find_components("Attention")
engine = db.get_engine_by_name("Transformer")
compatible = db.get_compatible_components(component_id, min_score=0.8)

# Compose new architectures
composer = EngineComposer(db)
components, score = composer.dream("greedy", start_component="SelectiveSSM")
print(f"Dreamed: {[c.name for c in components]} (score: {score:.2f})")

# Track benchmarks
from arcfusion import BenchmarkResult
result = BenchmarkResult(
    engine_id=engine.engine_id,
    benchmark_name="perplexity_wikitext",
    score=18.5,
    parameters={"model_size": "125M"}
)
db.add_benchmark(result)

# Leaderboard
leaderboard = db.get_benchmark_leaderboard("perplexity_wikitext", higher_is_better=False)
for engine, score in leaderboard:
    print(f"{engine.name}: {score}")

db.close()
```

## Database Schema

| Table | Purpose |
|-------|---------|
| `components` | Reusable building blocks (attention, FFN, etc.) |
| `engines` | Complete architectures (Transformer, Mamba, etc.) |
| `engine_components` | Links engines to their components |
| `component_relationships` | Component-to-component compatibility scores |
| `processed_papers` | Papers already analyzed (deduplication) |
| `benchmark_results` | Performance tracking |

## Seeded Architectures

- **Transformer** - The OG (Vaswani et al., 2017)
- **GPT-2** - Decoder-only with causal masking
- **BERT** - Bidirectional encoder
- **LLaMA** - RMSNorm + RoPE + SwiGLU
- **Mamba** - Selective State Space Model
- **RetNet** - Retention mechanism
- **RWKV** - Linear attention RNN
- **Mistral** - Grouped Query Attention

## Composition Strategies

| Strategy | Description |
|----------|-------------|
| `greedy` | Start with best component, add compatible ones |
| `random` | Random walk with temperature-controlled exploration |
| `mutate` | Swap components for compatible alternatives |
| `crossover` | Combine components from two parent engines |

## License

MIT
