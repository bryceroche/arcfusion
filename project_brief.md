# ArcFusion Project Brief

**Product Manager**: Claude (AI Agent)
**Reporting To**: Bryce Roche
**Project Status**: Active Development
**Last Updated**: 2025-12-11

---

## Mission Statement

Systematically harvest ML architecture knowledge from arXiv papers to build the most comprehensive component database for dreaming up novel neural architectures.

---

## Current State (Updated: 2025-12-11)

### Database Stats
| Metric | Value |
|--------|-------|
| Components | 21 |
| Engines | 211 |
| Relationships | 215 |
| Papers Processed | 203 |
| Benchmarks | 0 |

### What We Have (Completed)
- **21 components** - MultiHeadAttention, Embedding, LayerNorm, RMSNorm, FeedForward, RotaryEmbedding, ResidualConnection, SelectiveSSM, SwiGLU, GroupedQueryAttention, SoftmaxOutput, RetentionHead, CausalMask, TimeMixing, ChannelMixing, Activation, Gating, Convolution, Linear_Attention, Dropout, Normalization
- **211 engines** including landmark papers: Attention Is All You Need, BERT, GPT-3, Mamba, LLaMA 2, Mistral 7B, FlashAttention, ViT, Swin, Mixtral, Griffin, etc.
- **arXiv Fetcher** (`src/arcfusion/fetcher.py`) - Full paper fetching pipeline
- **CLI `arcfusion ingest`** - Batch ingestion command with search/ID modes
- **Composer system** with 4 strategies (greedy, random, mutate, crossover)

### What We Need (Next Phase)
- [ ] Run LLM analyzer on landmark papers to validate quality
- [ ] Benchmark extraction from papers
- [ ] Agent coordination for parallel processing

### New: LLM-Powered Deep Analysis
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

### 2025-12-11 - Phase 1 Complete
- Created project_brief.md
- Analyzed full codebase structure
- Implemented `src/arcfusion/fetcher.py` with ArxivFetcher class
- Added `arcfusion ingest` CLI command
- Ingested 203 papers including landmark architectures:
  - Attention Is All You Need (1706.03762)
  - BERT (1810.04805)
  - GPT-3 / Language Models are Few-Shot Learners (2005.14165)
  - Mamba (2312.00752)
  - LLaMA 2 (2307.09288)
  - Mistral 7B (2310.06825)
  - FlashAttention (2205.14135)
  - ViT (2010.11929)
  - Swin Transformer (2103.14030)
  - Mixtral of Experts (2401.04088)
  - Griffin (2402.19427)
  - And 190+ more research papers
- Discovered 6 new component categories through paper analysis
- Built 215 component-to-component relationships

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

---

## Next Actions

1. **Test LLM analyzer** - Run `arcfusion analyze --ids 2312.00752 1706.03762` with ANTHROPIC_API_KEY to validate component extraction quality
2. **More papers** - Continue ingestion to hit 500+ target
3. **Test composer** - Validate that enriched DB improves dream quality
4. **Benchmark extraction** - Parse performance metrics from papers
5. **PDF extraction** - Add pymupdf for full-text analysis (deeper than abstracts)

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
```

---

*Phase 1 complete. Ready for Phase 2: Scale & Quality.*
