"""
ArcFusion CLI - Command-line interface.
"""

import argparse
import sys
from . import __version__
from .db import ArcFusionDB
from .composer import EngineComposer
from .seeds import seed_transformers, seed_modern_architectures
from .fetcher import ArxivFetcher
from .dedup import ComponentDeduplicator, find_duplicate_engines
from .codegen import CodeGenerator


def cmd_init(args):
    """Initialize database with seed data."""
    db = ArcFusionDB(args.db)
    print(f"Initializing {args.db}...")
    print("\nSeeding Transformer components:")
    seed_transformers(db)
    print("\nSeeding modern architectures:")
    seed_modern_architectures(db)
    print("\nDone!")
    print(f"\nStats: {db.stats()}")
    db.close()


def cmd_stats(args):
    """Show database statistics."""
    db = ArcFusionDB(args.db)
    stats = db.stats()
    print(f"Database: {args.db}")
    print("-" * 40)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    db.close()


def cmd_list(args):
    """List components or engines."""
    db = ArcFusionDB(args.db)

    if args.type == "components":
        components = db.find_components()
        print(f"Components ({len(components)}):")
        for c in components:
            print(f"  [{c.component_id[:8]}] {c.name}: {c.usefulness_score:.2f}")

    elif args.type == "engines":
        engines = db.list_engines()
        print(f"Engines ({len(engines)}):")
        for e in engines:
            print(f"  [{e.engine_id[:8]}] {e.name}: {e.engine_score:.2f} ({len(e.component_ids)} components)")

    elif args.type == "papers":
        papers = db.list_processed_papers()
        print(f"Processed papers ({len(papers)}):")
        for p in papers:
            print(f"  [{p.arxiv_id}] {p.title} ({p.status})")

    elif args.type == "benchmarks":
        benchmarks = db.list_benchmarks()
        print(f"Benchmarks ({len(benchmarks)}):")
        for b in benchmarks:
            print(f"  {b['benchmark_name']}: {b['num_engines']} engines, avg={b['avg_score']:.2f}")

    db.close()


def cmd_dream(args):
    """Dream up a new architecture."""
    db = ArcFusionDB(args.db)
    composer = EngineComposer(db)

    kwargs = {}
    if args.strategy == "greedy" and args.start:
        kwargs["start_component"] = args.start
    elif args.strategy == "random":
        kwargs["steps"] = args.steps
        kwargs["temperature"] = args.temperature
    elif args.strategy == "mutate":
        kwargs["engine_name"] = args.engine
        kwargs["mutation_rate"] = args.rate
    elif args.strategy == "crossover":
        kwargs["engine1_name"] = args.engine1
        kwargs["engine2_name"] = args.engine2

    try:
        components, score = composer.dream(args.strategy, **kwargs)
    except ValueError as e:
        print(f"[ERROR] {e}")
        db.close()
        sys.exit(1)

    # Check for failure (score -1 indicates composition failed)
    if score < 0 or not components:
        print(f"[ERROR] Dream composition failed for strategy '{args.strategy}'")
        if args.strategy == "crossover":
            print(f"  Check that engines '{args.engine1}' and '{args.engine2}' exist")
            print(f"  Run: arcfusion list engines")
        elif args.strategy == "mutate":
            print(f"  Check that engine '{args.engine}' exists")
        db.close()
        sys.exit(1)

    print(f"Strategy: {args.strategy}")
    print(f"Estimated score: {score:.2f}")
    print(f"Components ({len(components)}):")
    for c in components:
        print(f"  - {c.name}")

    db.close()


def cmd_show(args):
    """Show details of a component or engine."""
    db = ArcFusionDB(args.db)

    # Try as engine first
    engine = db.get_engine_by_name(args.name)
    if engine:
        print(f"Engine: {engine.name}")
        print(f"  ID: {engine.engine_id}")
        print(f"  Score: {engine.engine_score}")
        print(f"  Paper: {engine.paper_url}")
        print(f"  Description: {engine.description[:200]}...")
        print(f"  Components ({len(engine.component_ids)}):")
        for cid in engine.component_ids:
            comp = db.get_component(cid)
            if comp:
                print(f"    - {comp.name}")
    else:
        # Try as component
        components = db.find_components(args.name)
        if components:
            comp = components[0]
            print(f"Component: {comp.name}")
            print(f"  ID: {comp.component_id}")
            print(f"  Score: {comp.usefulness_score}")
            print(f"  Description: {comp.description}")
            if comp.source_paper_id:
                print(f"  Source paper: {comp.source_paper_id} ({comp.introduced_year})")
            print(f"  Interface in: {comp.interface_in}")
            print(f"  Interface out: {comp.interface_out}")
            if comp.time_complexity:
                print(f"  Time complexity: {comp.time_complexity}")
            if comp.space_complexity:
                print(f"  Space complexity: {comp.space_complexity}")
            if comp.flops_formula:
                print(f"  FLOPs: {comp.flops_formula}")
            print(f"  Parallelizable: {comp.is_parallelizable}, Causal: {comp.is_causal}")
            if comp.math_operations:
                print(f"  Math ops: {', '.join(comp.math_operations)}")
            if comp.hyperparameters:
                print(f"  Hyperparameters: {comp.hyperparameters}")

            compatible = db.get_compatible_components(comp.component_id, min_score=0.7)
            if compatible:
                print(f"  Compatible with:")
                for cid, score in compatible[:5]:
                    c = db.get_component(cid)
                    if c:
                        print(f"    - {c.name}: {score:.2f}")
        else:
            print(f"Not found: {args.name}")

    db.close()


def cmd_ingest(args):
    """Ingest papers from arXiv."""
    db = ArcFusionDB(args.db)
    fetcher = ArxivFetcher(db)

    if args.query:
        # Custom search query
        print(f"Searching arXiv for: {args.query}")
        stats = fetcher.ingest_search(args.query, max_results=args.max)
    elif args.ids:
        # Specific paper IDs
        print(f"Fetching {len(args.ids)} specific papers...")
        papers = fetcher.fetch_by_ids(args.ids)
        stats = fetcher.ingest_batch(papers, max_papers=len(args.ids))
    else:
        # Default: search for architecture papers
        print(f"Searching for ML architecture papers in {args.category}...")
        stats = fetcher.ingest_architectures(
            max_results=args.max,
            category=args.category
        )

    # Show updated DB stats
    print(f"\nDatabase stats: {db.stats()}")
    db.close()


def cmd_analyze(args):
    """Deep LLM analysis of papers."""
    try:
        from .analyzer import PaperAnalyzer
    except ImportError:
        print("Error: anthropic package required for LLM analysis")
        print("Install with: pip install 'arcfusion[llm]'")
        sys.exit(1)

    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable required")
        sys.exit(1)

    db = ArcFusionDB(args.db)
    fetcher = ArxivFetcher(db)
    analyzer = PaperAnalyzer(db)

    total_new = 0
    for arxiv_id in args.ids:
        print(f"\n{'='*60}")
        print(f"Fetching {arxiv_id}...")

        paper = fetcher.fetch_by_id(arxiv_id)
        if not paper:
            print(f"  Could not fetch paper {arxiv_id}")
            continue

        engine, new_components = analyzer.analyze_and_ingest(
            title=paper.title,
            content=paper.abstract,
            paper_id=paper.arxiv_id,
            paper_url=paper.pdf_url,
            min_confidence=args.min_confidence,
        )

        total_new += len(new_components)

    print(f"\n{'='*60}")
    print(f"Analysis complete. Added {total_new} new components.")
    print(f"Database stats: {db.stats()}")
    db.close()


def cmd_dedup(args):
    """Find and merge duplicate components."""
    db = ArcFusionDB(args.db)
    deduplicator = ComponentDeduplicator(db)

    # Find duplicates
    print(f"Scanning for duplicates (threshold: {args.threshold})...\n")
    groups = deduplicator.find_duplicates(threshold=args.threshold)

    if not groups:
        print("No duplicates found!")
        db.close()
        return

    print(f"Found {len(groups)} duplicate groups:\n")
    print("=" * 60)

    for i, group in enumerate(groups, 1):
        print(f"\n{i}. KEEP: {group.canonical.name}")
        print(f"   (ID: {group.canonical.component_id[:8]}, score: {group.canonical.usefulness_score:.2f}, has_code: {bool(group.canonical.code.strip())})")
        print(f"   Reason: {group.similarity_reason}")
        print(f"   MERGE:")
        for dup in group.duplicates:
            print(f"     - {dup.name} ({dup.component_id[:8]}, score: {dup.usefulness_score:.2f})")

    # Also check for duplicate engines
    dup_engines = find_duplicate_engines(db)
    if dup_engines:
        print(f"\n{'=' * 60}")
        print(f"\nFound {len(dup_engines)} duplicate engine pairs:")
        for e1, e2, reason in dup_engines:
            print(f"  - {e1.name} ({e1.engine_id[:8]}) <-> {e2.name} ({e2.engine_id[:8]})")
            print(f"    Reason: {reason}")

    if args.dry_run:
        print(f"\n{'=' * 60}")
        print("\n[DRY RUN] No changes made. Use --apply to merge duplicates.")
    else:
        print(f"\n{'=' * 60}")
        print("\nMerging duplicates...")
        for group in groups:
            result = deduplicator.merge_group(group, dry_run=False)
            print(f"  Merged {len(result['merged'])} duplicates into '{result['canonical']}'")

        print(f"\nDone! Merged {sum(len(g.duplicates) for g in groups)} components.")
        print(f"Database stats: {db.stats()}")

    db.close()


def cmd_generate(args):
    """Generate PyTorch code from a dreamed architecture."""
    db = ArcFusionDB(args.db)
    gen = CodeGenerator(db)

    # Build kwargs for dream strategy
    kwargs = {}
    if args.strategy == "greedy" and args.start:
        kwargs["start_component"] = args.start
    elif args.strategy == "random":
        kwargs["steps"] = args.steps
        kwargs["temperature"] = args.temperature
    elif args.strategy == "mutate":
        kwargs["engine_name"] = args.engine
        kwargs["mutation_rate"] = args.rate
    elif args.strategy == "crossover":
        kwargs["engine1_name"] = args.engine1
        kwargs["engine2_name"] = args.engine2

    # Generate
    result = gen.generate_from_dream(args.strategy, name=args.name, **kwargs)

    # Validate
    valid, error = result.validate_syntax()
    if not valid:
        print(f"[ERROR] Generated code has syntax error: {error}")
        db.close()
        sys.exit(1)

    # Output
    if args.output:
        result.save(args.output)
        print(f"Generated {result.name} with {result.num_components} components")
        print(f"Saved to: {args.output}")
    else:
        print(result.code)

    print(f"\nComponents used:")
    for name in result.component_names:
        print(f"  - {name}")

    db.close()


def main():
    parser = argparse.ArgumentParser(
        description="ArcFusion - ML Architecture Component Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  arcfusion init                    Initialize with seed data
  arcfusion stats                   Show database statistics
  arcfusion list components         List all components
  arcfusion list engines            List all engines
  arcfusion show Transformer        Show engine details
  arcfusion show MultiHeadAttention Show component details
  arcfusion dream greedy --start Attention
  arcfusion dream random --steps 5
  arcfusion dream mutate --engine Transformer
  arcfusion dream crossover --engine1 BERT --engine2 Mamba
  arcfusion generate greedy -o model.py
  arcfusion generate crossover --engine1 BERT --engine2 Mamba -n HybridModel -o hybrid.py
"""
    )
    parser.add_argument("--db", default="arcfusion.db", help="Database path")
    parser.add_argument("--version", "-v", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # init
    subparsers.add_parser("init", help="Initialize database with seed data")

    # stats
    subparsers.add_parser("stats", help="Show database statistics")

    # list
    list_parser = subparsers.add_parser("list", help="List items")
    list_parser.add_argument("type", choices=["components", "engines", "papers", "benchmarks"])

    # show
    show_parser = subparsers.add_parser("show", help="Show details")
    show_parser.add_argument("name", help="Component or engine name")

    # dream
    dream_parser = subparsers.add_parser("dream", help="Dream up new architecture")
    dream_parser.add_argument("strategy", choices=["greedy", "random", "mutate", "crossover"])
    dream_parser.add_argument("--start", help="Starting component (greedy)")
    dream_parser.add_argument("--steps", type=int, default=5, help="Steps (random)")
    dream_parser.add_argument("--temperature", type=float, default=1.0, help="Temperature (random)")
    dream_parser.add_argument("--engine", help="Engine to mutate")
    dream_parser.add_argument("--rate", type=float, default=0.2, help="Mutation rate")
    dream_parser.add_argument("--engine1", help="First engine (crossover)")
    dream_parser.add_argument("--engine2", help="Second engine (crossover)")

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest papers from arXiv")
    ingest_parser.add_argument("--query", "-q", help="Custom arXiv search query")
    ingest_parser.add_argument("--ids", nargs="+", help="Specific arXiv IDs to fetch")
    ingest_parser.add_argument("--category", "-c", default="cs.LG", help="arXiv category (default: cs.LG)")
    ingest_parser.add_argument("--max", "-m", type=int, default=20, help="Max papers to ingest (default: 20)")

    # analyze (LLM-powered deep analysis)
    analyze_parser = subparsers.add_parser("analyze", help="LLM-powered component extraction (requires ANTHROPIC_API_KEY)")
    analyze_parser.add_argument("--ids", nargs="+", required=True, help="arXiv IDs to analyze")
    analyze_parser.add_argument("--min-confidence", type=float, default=0.7, help="Min confidence to add component (default: 0.7)")

    # dedup (find and merge duplicates)
    dedup_parser = subparsers.add_parser("dedup", help="Find and merge duplicate components")
    dedup_parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold (0-1, default: 0.5)")
    dedup_parser.add_argument("--apply", dest="dry_run", action="store_false", default=True, help="Apply changes (default is dry-run)")

    # generate (code generation from dream)
    gen_parser = subparsers.add_parser("generate", help="Generate PyTorch code from dreamed architecture")
    gen_parser.add_argument("strategy", choices=["greedy", "random", "mutate", "crossover"])
    gen_parser.add_argument("--name", "-n", default="DreamedArchitecture", help="Class name for generated architecture")
    gen_parser.add_argument("--output", "-o", help="Output file path (prints to stdout if not specified)")
    gen_parser.add_argument("--start", help="Starting component (greedy)")
    gen_parser.add_argument("--steps", type=int, default=6, help="Number of components (random)")
    gen_parser.add_argument("--temperature", type=float, default=1.0, help="Temperature (random)")
    gen_parser.add_argument("--engine", help="Engine to mutate")
    gen_parser.add_argument("--rate", type=float, default=0.2, help="Mutation rate")
    gen_parser.add_argument("--engine1", help="First engine (crossover)")
    gen_parser.add_argument("--engine2", help="Second engine (crossover)")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "show":
        cmd_show(args)
    elif args.command == "dream":
        cmd_dream(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "dedup":
        cmd_dedup(args)
    elif args.command == "generate":
        cmd_generate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
