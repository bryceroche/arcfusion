# ArcFusion Project Brief

**Product Manager**: Claude (AI Agent)
**Reporting To**: Bryce Roche
**Project Status**: Active Development
**Last Updated**: 2025-12-16

---

## Mission Statement

Systematically harvest ML architecture knowledge from arXiv papers to build the most comprehensive component database for dreaming up novel neural architectures.

---

## Current State (Updated: 2025-12-16)

### Database Stats
| Metric | Value |
|--------|-------|
| Components | 60+ |
| Engines | 9 (Transformer, BERT, LLaMA, Mistral-7B, GPT-2, RWKV, RetNet, Mamba, FlashAttention) |
| Relationships | 130 |
| Training Runs | 48 |
| Best PPL | 217.1 (MHA32) |

### Cloud Training Pipeline (NEW - 2025-12-16)
- **Modal A100 GPU** - `scripts/cloud_train_fair.py` orchestrates remote training
- **Model Templates** - `src/arcfusion/model_templates.py` generates GQA/MQA/MHA/Mamba code
- **Dream & Train** - `scripts/dream_and_train.py` automates architecture exploration
- **Results DB** - `training_runs` + `training_insights` tables track all experiments
- **Key findings**: 32L optimal depth, GQA14 best efficiency, MambaFast 4.64x speedup

### What We Have (Completed)

**Core Pipeline:**
- **60+ components** across 8 categories (attention, structure, layer, position, embedding, training, efficiency, output)
- **9 curated engines** with full component relationships
- **arXiv Fetcher** (`src/arcfusion/fetcher.py`) - Full paper fetching pipeline
- **CLI `arcfusion ingest`** - Batch ingestion command with search/ID modes

**Composer System:**
- 4 dream strategies: greedy, random, mutate, crossover
- Recipe system for Composer → ML Agent handoff
- RecipeAdjustment tracking for training modifications

**Training Pipeline:**
- Modal A100 GPU training with automatic result logging
- Model templates for GQA/MQA/MHA/Mamba architectures
- Auto-generated insights from training results

**Code Generation:**
- **`src/arcfusion/codegen.py`** - Generates runnable PyTorch nn.Module code
- Handles all component categories: embedding, attention, layer, position, output
- Auto-generates residual connections and proper shape handling

### What We Need (Next Phase)
- [ ] ML Agent implementation (recipe executor with adjustment tracking)
- [ ] Cloud training integration (Groq/Modal/Lambda)
- [ ] Scale up paper ingestion (target: 500+ papers)

### LLM-Powered Deep Analysis
- **`src/arcfusion/analyzer.py`** - Claude-powered paper analysis
- Extracts specific component variants (not generic categories)
- Captures: interfaces, hyperparameters, complexity, math ops, code sketches
- CLI: `arcfusion analyze --ids <arxiv_ids>` (requires ANTHROPIC_API_KEY)

---

## Goals

### Phase 1: Paper Ingestion Pipeline (Current)
1. **Fetch papers from arXiv** - Use arXiv API or RSS feeds
2. **Extract text** - Abstract + full paper when available
3. **Decompose into components** - Use existing `PaperDecomposer`
4. **Populate database** - New components, engines, relationships

### Phase 2: Scale & Quality
- Process 100+ papers per session
- Improve component extraction accuracy
- Build richer c2c_score relationships
- Add benchmark data from papers

### Phase 3: Novel Architectures
- Use enriched database for better `dream` compositions
- Generate and evaluate novel architecture candidates
- Track which dreamed architectures get validated

---

## Agent Delegation Structure

```
You (Bryce) - Executive
    |
    v
PM Agent (Claude) - Strategy, Coordination, Reporting
    |
    +---> Explore Agent - Codebase navigation, file discovery
    |
    +---> Research Agent - arXiv paper fetching, reading
    |
    +---> Implementation Agent - Code writing, bug fixes
    |
    +---> QA Agent - Testing, validation
```

### Agent Responsibilities

**PM Agent (Me)**
- Maintain project brief and roadmap
- Delegate tasks to specialized agents
- Track progress and report to you
- Make architectural decisions
- Prioritize work

**Explore Agent**
- Navigate codebase structure
- Find relevant files and patterns
- Answer questions about existing code

**Research Agent**
- Fetch arXiv papers (abstracts, PDFs)
- Extract paper metadata
- Identify high-value papers to process

**Implementation Agent**
- Write new code (fetching, parsing, etc.)
- Fix bugs and issues
- Add new features

**QA Agent**
- Run tests
- Validate data quality
- Check for regressions

---

## Immediate Priorities

### P0 - Critical Path
1. **arXiv Fetcher** - Fetch papers by category/search query
2. **Text Extractor** - Get abstract + body text from papers
3. **Batch Processor** - Process multiple papers in sequence

### P1 - Important
4. **Quality Scoring** - Better confidence for extracted components
5. **Relationship Mining** - Extract c2c relationships from papers
6. **CLI Integration** - `arcfusion ingest` command

### P2 - Nice to Have
7. **RSS Feed Monitoring** - Auto-ingest new papers
8. **Benchmark Extraction** - Parse results tables from papers
9. **Citation Graph** - Track paper lineage

---

## Success Metrics

| Metric | Start | Current | Target | Status |
|--------|-------|---------|--------|--------|
| Components in DB | 15 | 21 | 100+ | 21% |
| Engines in DB | 8 | 211 | 50+ | EXCEEDED |
| Papers processed | 0 | 203 | 500+ | 41% |
| C2C relationships | 15 | 215 | 500+ | 43% |
| Avg composition score | ~0.85 | TBD | 0.90+ | - |

---

## Technical Approach

### arXiv Integration Options

**Option A: arXiv API (Recommended)**
```python
import arxiv

# Search for papers
search = arxiv.Search(
    query="cat:cs.LG AND (transformer OR attention OR state space)",
    max_results=100,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

for paper in search.results():
    title = paper.title
    abstract = paper.summary
    arxiv_id = paper.entry_id.split('/')[-1]
    pdf_url = paper.pdf_url
```

**Option B: RSS Feeds**
- `http://export.arxiv.org/rss/cs.LG` - Machine Learning
- `http://export.arxiv.org/rss/cs.CL` - Computation and Language
- `http://export.arxiv.org/rss/cs.CV` - Computer Vision

### Text Extraction
- **Abstract**: Direct from arXiv API
- **Full text**: Download PDF, use `pymupdf` or `pdfplumber`

---

## Session Log

### 2025-12-12 - Recipe System & Validation Complete
- **Recipe dataclass** - Composer → ML Agent handoff format with assembly instructions
- **RecipeAdjustment** - Tracks ML Agent modifications during training
- **DB tables** - recipes, recipe_adjustments with full CRUD operations
- **Composer.create_recipe()** - Generates recipes from any dream strategy
- **Validation pipeline working** - Transformer builds and trains from 7 DB components
- Tests: 199 tests passing
- Closes: arcfusion-p9w, arcfusion-9c9
- Unblocks: arcfusion-a1s (ML Agent implementation)

### 2025-12-11 - Validation Pipeline
- Created `src/arcfusion/validator.py` - Auto-validation of dreamed architectures
- ModelBuilder compiles generated code into PyTorch models
- TrainingHarness trains models on synthetic data
- BenchmarkRunner computes perplexity and metrics
- Fixed codegen embedding/output handling for vocab_size
- Tests: 183 tests passing

### 2025-12-11 - LLM Analyzer Complete
- Created `src/arcfusion/analyzer.py` - Claude-powered deep paper analysis
- Extracts specific component variants (not generic categories like "attention")
- Captures full metadata: interfaces, hyperparameters, complexity, FLOPs, math ops
- Generates PyTorch code sketches for each component
- Tracks innovation and component lineage (builds_on)
- Added confidence filtering (default min 0.7)
- Added `arcfusion analyze` CLI command
- Tests: 32 tests passing (9 new analyzer tests)
- Optional dependency: `pip install 'arcfusion[llm]'` for anthropic package

### 2025-12-11 - Phase 1 Complete
- Created project_brief.md
- Analyzed full codebase structure
- Implemented `src/arcfusion/fetcher.py` with ArxivFetcher class
- Added `arcfusion ingest` CLI command
- Ingested 203 papers including landmark architectures
- Discovered 6 new component categories through paper analysis
- Built 215 component-to-component relationships

---

## Next Actions

1. **ML Agent implementation** - Execute recipes with modification tracking (arcfusion-a1s)
2. **Cloud training integration** - Groq/Modal/Lambda for auto-pipeline (arcfusion-zzp)
3. **Scale up paper ingestion** - Hit 500+ papers target (arcfusion-xol)
4. **Component granularity** - Ensure distinct, trainable recipes (arcfusion-wbi)
5. **Test LLM analyzer** - Run on landmark papers to validate extraction quality

---

## CLI Commands

```bash
# Initialize with seed data
arcfusion init

# Ingest papers by search
arcfusion ingest --query "cat:cs.LG AND mamba" --max 50

# Ingest specific papers
arcfusion ingest --ids 1706.03762 2312.00752

# Ingest architecture papers (default)
arcfusion ingest --max 30

# Deep LLM analysis (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
arcfusion analyze --ids 2312.00752 1706.03762  # Mamba, Transformer
arcfusion analyze --ids 1810.04805 --min-confidence 0.8  # BERT

# Dream up new architectures
arcfusion dream greedy --start MultiHeadAttention
arcfusion dream crossover --engine1 Transformer --engine2 Mamba

# Generate code from dreamed architecture
arcfusion dream greedy --codegen --output dreamed_arch.py
```

## Python API - Recipe System

```python
from arcfusion import ArcFusionDB, Recipe, RecipeAdjustment
from arcfusion.composer import EngineComposer

# Create a recipe from dreamed components
db = ArcFusionDB("arcfusion.db")
composer = EngineComposer(db)

# Dream and create recipe with assembly instructions
recipe = composer.create_recipe(
    name="NovelTransformer",
    strategy="crossover",
    engine1_name="Transformer",
    engine2_name="Mamba"
)

# Recipe contains everything ML Agent needs
print(recipe.component_ids)  # Ordered list
print(recipe.assembly)       # connections, residuals, shapes, notes
print(recipe.strategy)       # How it was created

# Record adjustments made during training
adjustment = RecipeAdjustment(
    recipe_id=recipe.recipe_id,
    adjustment_type="shape_fix",
    original_value="d_model=512",
    adjusted_value="d_model=128",
    reason="Reduced for memory constraints"
)
db.add_adjustment(adjustment)
```

---

*Validation pipeline complete. Recipe system ready. Next: ML Agent implementation.*
