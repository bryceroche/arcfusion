# Claude Integration for ArcFusion

**Note**: This project uses [bd (beads)](https://github.com/steveyegge/beads) for issue tracking. Use `bd` commands instead of markdown TODOs. See AGENTS.md for workflow details.

## Project Context

ArcFusion is an ML architecture component database. Claude is used for:

1. **Paper Analysis** (`analyzer.py`) - Extract components from arXiv papers
2. **Code Assistance** - Development help via Claude Code

## LLM Analysis Integration

### How It Works
The `PaperAnalyzer` class in `src/arcfusion/analyzer.py` uses Claude to:
- Extract architectural components from paper abstracts
- Identify interfaces, hyperparameters, complexity
- Generate PyTorch code sketches
- Score confidence for each component

### API Usage
```python
from arcfusion.analyzer import PaperAnalyzer
from arcfusion.db import ArcFusionDB

db = ArcFusionDB("arcfusion.db")
analyzer = PaperAnalyzer(db)  # Uses ANTHROPIC_API_KEY env var

engine, components = analyzer.analyze_and_ingest(
    title="Paper Title",
    content="Abstract text...",
    paper_id="2312.00752",
    paper_url="https://arxiv.org/pdf/2312.00752",
    min_confidence=0.7
)
```

### Analysis Prompt Structure
The analyzer uses a detailed prompt that asks Claude to extract:
- Component name, category, description
- Input/output interfaces with tensor shapes
- Hyperparameters with values from paper
- Time/space complexity
- Math operations
- Whether parallelizable/causal
- What makes it novel
- PyTorch code sketch

### Component Categories
- `attention` - Attention mechanisms (self, cross, multi-head, SSM)
- `structure` - Encoder/decoder blocks, stacks
- `layer` - FFN, normalization, activation, dropout
- `position` - Positional encodings (sinusoidal, RoPE, learned)
- `embedding` - Token embeddings
- `training` - Optimizers, LR schedules, loss functions
- `efficiency` - FlashAttention, KV-cache, quantization
- `output` - Output projections, heads

## Claude Code Workflow

### Issue Tracking with bd
```bash
# Check what's ready to work on
bd ready --json

# Create a new issue
bd create "Add support for GPT-4 paper analysis" -t feature -p 2 --json

# Claim an issue
bd update arcfusion-1 --status in_progress --json

# Complete an issue
bd close arcfusion-1 --reason "Implemented GPT-4 paper parsing" --json
```

### Development Commands
```bash
# Run tests
python3 -m pytest tests/ -v

# Test specific module
python3 -m pytest tests/test_analyzer.py -v

# Run CLI
python3 -m arcfusion.cli stats
python3 -m arcfusion.cli dream greedy

# Analyze a paper (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY="sk-ant-..."
python3 -m arcfusion.cli analyze --ids 1706.03762
```

### Key Files to Know
| File | Purpose |
|------|---------|
| `src/arcfusion/db.py` | Database schema, CRUD operations |
| `src/arcfusion/analyzer.py` | LLM paper analysis |
| `src/arcfusion/composer.py` | Dream engine strategies |
| `src/arcfusion/dedup.py` | Fuzzy deduplication |
| `src/arcfusion/cli.py` | Command-line interface |
| `tests/test_analyzer.py` | Analyzer tests (no API calls) |

### Testing Without API Key
The test suite runs without requiring an API key:
```python
# tests/test_analyzer.py tests dataclass parsing and DB integration
# without making actual API calls
pytest tests/test_analyzer.py -v  # All pass without ANTHROPIC_API_KEY
```

## Common Tasks

### Add a New Architecture
```bash
# 1. Analyze the paper
arcfusion analyze --ids <arxiv_id>

# 2. Check for duplicates
arcfusion dedup

# 3. Apply dedup if needed
arcfusion dedup --apply

# 4. Verify
arcfusion list engines
arcfusion show <engine_name>
```

### Dream a New Architecture
```bash
# Greedy composition from best components
arcfusion dream greedy

# Random walk exploration
arcfusion dream random --steps 8 --temperature 0.8

# Crossover two architectures
arcfusion dream crossover --engine1 BERT --engine2 Mamba

# Mutate an existing architecture
arcfusion dream mutate --engine Transformer --rate 0.3
```

### Debug Database Issues
```python
from src.arcfusion.db import ArcFusionDB

db = ArcFusionDB("arcfusion.db")
print(db.stats())

# Check components
for c in db.find_components()[:5]:
    print(f"{c.name}: {c.interface_in} -> {c.interface_out}")

# Check relationships
rows = db.conn.execute(
    "SELECT * FROM component_relationships LIMIT 5"
).fetchall()
```

## Best Practices

1. **Always run tests** before committing: `pytest tests/ -v`
2. **Use bd for tracking** - no markdown TODOs
3. **Check dedup** after adding new components
4. **Verify interfaces** when debugging composition issues
5. **Commit .beads/** files with code changes
6. **Land the plane** - Always push to remote at session end (see AGENTS.md)

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Required for `arcfusion analyze` command |

## Links

- [AGENTS.md](./AGENTS.md) - Full agent workflow documentation
- [beads](https://github.com/steveyegge/beads) - Issue tracker
- [Anthropic API](https://docs.anthropic.com/) - Claude API docs
