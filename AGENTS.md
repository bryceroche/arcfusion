# ArcFusion - AI Agent Instructions

## Project Overview

**ArcFusion** is a machine learning architecture component database that extracts, stores, and recombines architectural components from research papers to "dream up" novel architectures.

### Tech Stack
- **Language**: Python 3.14+
- **Database**: SQLite (arcfusion.db)
- **LLM Integration**: Anthropic Claude API
- **Testing**: pytest
- **Package Manager**: pip/uv

### Project Structure
```
arcfusion/
├── src/arcfusion/
│   ├── db.py          # Database layer (788 lines)
│   ├── seeds.py       # Seed data for architectures (718 lines)
│   ├── composer.py    # Dream engine strategies (440 lines)
│   ├── dedup.py       # Fuzzy deduplication (415 lines)
│   ├── analyzer.py    # LLM component extraction (373 lines)
│   ├── cli.py         # CLI commands (362 lines)
│   ├── fetcher.py     # arXiv paper fetching (282 lines)
│   └── decomposer.py  # Pattern-based extraction (108 lines)
├── tests/
│   ├── test_analyzer.py
│   ├── test_db.py
│   └── test_fetcher.py
├── .beads/            # Issue tracking
└── arcfusion.db       # SQLite database
```

### Key Commands
```bash
# Development
python3 -m pytest tests/ -v          # Run tests (32 tests)
python3 -m arcfusion.cli <command>   # Run CLI

# CLI Commands
arcfusion init                       # Seed database
arcfusion stats                      # Show statistics
arcfusion list components|engines    # List items
arcfusion show <name>                # Component/engine details
arcfusion dream greedy|random|crossover|mutate
arcfusion ingest --query "..."       # Fetch from arXiv
arcfusion analyze --ids <arxiv_id>   # Deep LLM analysis
arcfusion dedup [--apply]            # Find/merge duplicates
```

### Database State
- **60 components** across 8 categories (layer, structure, attention, training, output, embedding, position, efficiency)
- **9 engines**: Transformer, BERT, LLaMA, Mistral-7B, GPT-2, RWKV, RetNet, Mamba, FlashAttention
- **130 component relationships**

---

## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Auto-syncs to JSONL for version control
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**
```bash
bd ready --json
```

**Create new issues:**
```bash
bd create "Issue title" -t bug|feature|task -p 0-4 --json
bd create "Issue title" -p 1 --deps discovered-from:arcfusion-123 --json
bd create "Subtask" --parent <epic-id> --json
```

**Claim and update:**
```bash
bd update arcfusion-42 --status in_progress --json
bd update arcfusion-42 --priority 1 --json
```

**Complete work:**
```bash
bd close arcfusion-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `bd ready` shows unblocked issues
2. **Claim your task**: `bd update <id> --status in_progress`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `bd create "Found bug" -p 1 --deps discovered-from:<parent-id>`
5. **Complete**: `bd close <id> --reason "Done"`
6. **Land the plane**: See session end checklist below

### Land the Plane (Session End)

**CRITICAL**: A session is not complete until all work is pushed to remote. Follow this checklist:

1. **File remaining work** - Create bd issues for any incomplete tasks
2. **Run quality gates** - If code changed: `pytest tests/ -v`
   - File P0 issues for any failures
3. **Update issue statuses** - Close finished work, update in-progress items
4. **Push sequence**:
   ```bash
   git pull --rebase
   bd sync                    # Flush pending changes
   git add -A && git commit -m "message"
   git push                   # MANDATORY - never skip
   git status                 # Verify "up to date with origin"
   ```
5. **Propose next work** - Suggest follow-up issues for future sessions

**Rules**:
- **Never stop before pushing** - Unpushed work breaks multi-agent coordination
- **Never say "ready to push when you are"** - The agent must push
- **Verify success** - Confirm `git push` completes before ending

See full details: [beads AGENT_INSTRUCTIONS.md](https://github.com/steveyegge/beads/blob/main/AGENT_INSTRUCTIONS.md)

### Important Rules

- Use bd for ALL task tracking
- Always use `--json` flag for programmatic use
- Link discovered work with `discovered-from` dependencies
- Check `bd ready` before asking "what should I work on?"
- Do NOT create markdown TODO lists
- Do NOT duplicate tracking systems

---

## Project-Specific Guidelines

### Adding New Components
1. Use `arcfusion analyze --ids <arxiv_id>` for LLM extraction
2. Run `arcfusion dedup` to check for duplicates before applying
3. Verify interfaces are properly specified (shape, dtype)

### Dream Composition
Four strategies available:
- **greedy**: Best-first compatible component selection
- **random**: Temperature-controlled exploration
- **crossover**: Category-wise parent combination
- **mutate**: Interface-compatible component swapping

### Testing Requirements
- All tests must pass: `pytest tests/ -v`
- Add tests for new functionality in appropriate test file
- Use temporary databases for test fixtures

### Code Style
- Use dataclasses for models
- JSON serialization for complex fields
- Type hints on all functions
- Docstrings for public functions

---

## Current Development Focus

### Completed
- [x] Core database schema with 8 tables
- [x] LLM-powered deep analysis (Claude API)
- [x] Fuzzy deduplication with variant detection
- [x] Interface-aware dream composition
- [x] CLI with 8 commands

### Future Enhancements
- [ ] Benchmark integration for scoring
- [ ] Code generation from dreamed architectures
- [ ] Web UI for exploration
- [ ] More papers in knowledge base
