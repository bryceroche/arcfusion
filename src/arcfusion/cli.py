"""
ArcFusion CLI - Command-line interface.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

# Load .env file from current directory or project root
load_dotenv()

from . import __version__
from .db import ArcFusionDB, BenchmarkResult, generate_training_insights
from .composer import EngineComposer
from .seeds import seed_transformers, seed_modern_architectures
from .fetcher import ArxivFetcher
from .dedup import ComponentDeduplicator, find_duplicate_engines
from .codegen import CodeGenerator

# Conditional import for validator (requires torch)
try:
    from .validator import ValidationPipeline, ModelConfig, TrainingConfig
    HAS_VALIDATOR = True
except ImportError:
    HAS_VALIDATOR = False

# Conditional import for cloud training (requires modal)
try:
    from .cloud import CloudTrainer, CloudConfig
    HAS_CLOUD = True
except ImportError:
    HAS_CLOUD = False

# Conditional import for web UI (requires fastapi/uvicorn)
try:
    from .web import run_server, HAS_FASTAPI
    HAS_WEB = HAS_FASTAPI
except ImportError:
    HAS_WEB = False

# CLI display constants
SEPARATOR_WIDTH = 60
SEPARATOR = "-" * SEPARATOR_WIDTH
SECTION_SEPARATOR = "=" * SEPARATOR_WIDTH


def _cli_error(msg: str, exit_code: int = 1) -> None:
    """Print error message and exit."""
    print(f"[ERROR] {msg}")
    sys.exit(exit_code)


def _build_dream_kwargs(args: argparse.Namespace) -> dict:
    """Build kwargs dict for dream/compose strategies from CLI args."""
    kwargs = {}
    if args.strategy == "greedy" and getattr(args, 'start', None):
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
    return kwargs


def _add_dream_strategy_args(parser: argparse.ArgumentParser, random_steps: int = 5) -> None:
    """Add common dream strategy arguments to a parser."""
    parser.add_argument("--start", help="Starting component name for greedy strategy")
    parser.add_argument("--steps", type=int, default=random_steps, help=f"Number of components for random strategy (default: {random_steps})")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for random strategy (default: 1.0)")
    parser.add_argument("--engine", help="Engine name to mutate (required for mutate strategy)")
    parser.add_argument("--rate", type=float, default=0.2, help="Mutation rate 0-1 (default: 0.2)")
    parser.add_argument("--engine1", help="First parent engine (required for crossover)")
    parser.add_argument("--engine2", help="Second parent engine (required for crossover)")


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize database with seed data."""
    with ArcFusionDB(args.db) as db:
        print(f"Initializing {args.db}...")
        print("\nSeeding Transformer components:")
        seed_transformers(db)
        print("\nSeeding modern architectures:")
        seed_modern_architectures(db)
        print("\nDone!")
        print(f"\nStats: {db.stats()}")


def cmd_stats(args: argparse.Namespace) -> None:
    """Show database statistics."""
    with ArcFusionDB(args.db) as db:
        stats = db.stats()
        print(f"Database: {args.db}")
        print("-" * 40)
        for k, v in stats.items():
            print(f"  {k}: {v}")

        # Show surrogate model accuracy if available
        accuracy = db.get_surrogate_accuracy_stats()
        if accuracy.get('n_samples', 0) > 0:
            print()
            print("Surrogate Model Accuracy:")
            print("-" * 40)
            if accuracy.get('insufficient_data'):
                print(f"  samples: {accuracy['n_samples']} (need 2+ for metrics)")
            else:
                print(f"  samples: {accuracy['n_samples']}")
                print(f"  PPL MAE: {accuracy['ppl_mae']:.1f}")
                print(f"  PPL MAPE: {accuracy['ppl_mape']:.1f}%")
                print(f"  PPL correlation: {accuracy['ppl_correlation']:.3f}")
                if accuracy.get('time_mae') is not None:
                    print(f"  Time MAE: {accuracy['time_mae']:.1f}s")
                    print(f"  Time MAPE: {accuracy['time_mape']:.1f}%")
                if accuracy.get('time_correlation') is not None:
                    print(f"  Time correlation: {accuracy['time_correlation']:.3f}")


def cmd_list(args: argparse.Namespace) -> None:
    """List components or engines."""
    with ArcFusionDB(args.db) as db:
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


def cmd_dream(args: argparse.Namespace) -> None:
    """Dream up a new architecture."""
    with ArcFusionDB(args.db) as db:
        composer = EngineComposer(db)
        kwargs = _build_dream_kwargs(args)

        # Use create_recipe if saving, otherwise just dream
        if getattr(args, 'save', False):
            recipe_name = getattr(args, 'name', None) or f"Dream_{args.strategy}"
            try:
                recipe = composer.create_recipe(
                    name=recipe_name,
                    strategy=args.strategy,
                    save_to_db=True,
                    **kwargs
                )
            except ValueError as e:
                _cli_error(str(e))

            print(f"Strategy: {args.strategy}")
            print(f"Estimated score: {recipe.estimated_score:.2f}")
            print(f"Recipe ID: {recipe.recipe_id}")
            print(f"Components ({len(recipe.component_ids)}):")
            for cid in recipe.component_ids:
                comp = db.get_component(cid)
                if comp:
                    print(f"  - {comp.name}")
            print(f"\n✓ Saved recipe '{recipe_name}' to database")
        else:
            try:
                components, score = composer.dream(args.strategy, **kwargs)
            except ValueError as e:
                _cli_error(str(e))

            # Check for failure (score -1 indicates composition failed)
            if score < 0 or not components:
                msg = f"Dream composition failed for strategy '{args.strategy}'"
                if args.strategy == "crossover":
                    msg += f"\n  Check that engines '{args.engine1}' and '{args.engine2}' exist"
                elif args.strategy == "mutate":
                    msg += f"\n  Check that engine '{args.engine}' exists"
                _cli_error(msg)

            print(f"Strategy: {args.strategy}")
            print(f"Estimated score: {score:.2f}")
            print(f"Components ({len(components)}):")
            for c in components:
                print(f"  - {c.name}")
            print("\n(Use --save to persist this recipe)")


def cmd_show(args: argparse.Namespace) -> None:
    """Show details of a component or engine."""
    with ArcFusionDB(args.db) as db:
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
                    print("  Compatible with:")
                    for cid, score in compatible[:5]:
                        c = db.get_component(cid)
                        if c:
                            print(f"    - {c.name}: {score:.2f}")
            else:
                print(f"Not found: {args.name}")


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest papers from arXiv."""
    with ArcFusionDB(args.db) as db:
        fetcher = ArxivFetcher(db)

        if args.query:
            # Custom search query
            print(f"Searching arXiv for: {args.query}")
            fetcher.ingest_search(args.query, max_results=args.max)
        elif args.ids:
            # Specific paper IDs
            print(f"Fetching {len(args.ids)} specific papers...")
            papers = fetcher.fetch_by_ids(args.ids)
            fetcher.ingest_batch(papers, max_papers=len(args.ids))
        else:
            # Default: search for architecture papers
            print(f"Searching for ML architecture papers in {args.category}...")
            fetcher.ingest_architectures(
                max_results=args.max,
                category=args.category
            )

        # Show updated DB stats
        print(f"\nDatabase stats: {db.stats()}")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Deep LLM analysis of papers."""
    try:
        from .analyzer import PaperAnalyzer
    except ImportError:
        print("[ERROR] anthropic package required for LLM analysis")
        print("  Install with: pip install 'arcfusion[llm]'")
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[ERROR] ANTHROPIC_API_KEY not set")
        print("  Set via environment: export ANTHROPIC_API_KEY=sk-ant-...")
        print("  Or add to .env file: ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    with ArcFusionDB(args.db) as db:
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


def cmd_dedup(args: argparse.Namespace) -> None:
    """Find and merge duplicate components."""
    with ArcFusionDB(args.db) as db:
        deduplicator = ComponentDeduplicator(db)

        # Find duplicates
        print(f"Scanning for duplicates (threshold: {args.threshold})...\n")
        groups = deduplicator.find_duplicates(threshold=args.threshold)

        if not groups:
            print("No duplicates found!")
            return

        print(f"Found {len(groups)} duplicate groups:\n")
        print(SECTION_SEPARATOR)

        for i, group in enumerate(groups, 1):
            print(f"\n{i}. KEEP: {group.canonical.name}")
            print(f"   (ID: {group.canonical.component_id[:8]}, score: {group.canonical.usefulness_score:.2f}, has_code: {bool(group.canonical.code.strip())})")
            print(f"   Reason: {group.similarity_reason}")
            print("   MERGE:")
            for dup in group.duplicates:
                print(f"     - {dup.name} ({dup.component_id[:8]}, score: {dup.usefulness_score:.2f})")

        # Also check for duplicate engines
        dup_engines = find_duplicate_engines(db)
        if dup_engines:
            print(f"\n{SECTION_SEPARATOR}")
            print(f"\nFound {len(dup_engines)} duplicate engine pairs:")
            for e1, e2, reason in dup_engines:
                print(f"  - {e1.name} ({e1.engine_id[:8]}) <-> {e2.name} ({e2.engine_id[:8]})")
                print(f"    Reason: {reason}")

        if args.dry_run:
            print(f"\n{SECTION_SEPARATOR}")
            print("\n[DRY RUN] No changes made. Use --apply to merge duplicates.")
        else:
            print(f"\n{SECTION_SEPARATOR}")
            print("\nMerging duplicates...")
            for group in groups:
                result = deduplicator.merge_group(group, dry_run=False)
                print(f"  Merged {len(result['merged'])} duplicates into '{result['canonical']}'")

            print(f"\nDone! Merged {sum(len(g.duplicates) for g in groups)} components.")
            print(f"Database stats: {db.stats()}")


def _config_list(db: ArcFusionDB, args: argparse.Namespace) -> None:
    """List all configurations."""
    configs = db.find_configurations(
        min_score=args.min_score,
        validated=True if args.validated else None
    )
    if not configs:
        print("No configurations found.")
        return

    print(f"Configurations ({len(configs)}):")
    for config in configs:
        status = "✓" if config.validated else " "
        print(f"  [{status}] {config.name}")
        print(f"      Score: {config.config_score:.2f}, Used: {config.usage_count}x")
        print(f"      Components: {len(config.component_ids)}")


def _config_extract(db: ArcFusionDB, composer: EngineComposer, args: argparse.Namespace) -> None:
    """Extract configurations from an engine."""
    if not args.engine:
        _cli_error("--engine required for extract action")

    configs = composer.extract_configurations_from_engine(
        args.engine,
        min_size=args.min_size,
        max_size=args.max_size
    )

    if not configs:
        print(f"No configurations extracted from '{args.engine}'")
        return

    print(f"Extracted {len(configs)} configurations from '{args.engine}':")
    for config in configs[:10]:  # Show first 10
        print(f"  - {config.name} (score: {config.config_score:.2f})")

    if len(configs) > 10:
        print(f"  ... and {len(configs) - 10} more")

    if not args.dry_run:
        saved = composer.save_configurations(configs)
        print(f"\nSaved {saved} new configurations to database.")
    else:
        print("\n[DRY RUN] Use --save to persist configurations.")


def _config_show(db: ArcFusionDB, args: argparse.Namespace) -> None:
    """Show details of a configuration."""
    if not args.config_id:
        _cli_error("--id required for show action")

    config = db.get_configuration(args.config_id)
    if not config:
        _cli_error(f"Configuration '{args.config_id}' not found")

    print(f"Configuration: {config.name}")
    print(f"  ID: {config.config_id}")
    print(f"  Score: {config.config_score:.2f}")
    print(f"  Usage count: {config.usage_count}")
    print(f"  Validated: {'Yes' if config.validated else 'No'}")
    print(f"  Source engine: {config.source_engine_id or 'N/A'}")
    print(f"  Description: {config.description}")
    print(f"\nComponents ({len(config.component_ids)}):")
    for cid in config.component_ids:
        comp = db.get_component(cid)
        if comp:
            print(f"    - {comp.name}")
        else:
            print(f"    - [Unknown: {cid}]")


def cmd_config(args: argparse.Namespace) -> None:
    """Manage component configurations."""
    with ArcFusionDB(args.db) as db:
        composer = EngineComposer(db)

        if args.action == "list":
            _config_list(db, args)
        elif args.action == "extract":
            _config_extract(db, composer, args)
        elif args.action == "show":
            _config_show(db, args)


def _benchmark_add(db: ArcFusionDB, args: argparse.Namespace) -> None:
    """Add a benchmark result."""
    if not args.engine or not args.name or args.score is None:
        _cli_error("--engine, --name, and --score required for add action")

    engine = db.get_engine_by_name(args.engine)
    if not engine:
        _cli_error(f"Engine '{args.engine}' not found")

    # Parse parameters if provided
    parameters = {}
    if args.params:
        for param in args.params:
            if "=" in param:
                key, value = param.split("=", 1)
                try:
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    pass
                parameters[key] = value

    result = BenchmarkResult(
        engine_id=engine.engine_id,
        benchmark_name=args.name,
        score=args.score,
        parameters=parameters,
        notes=args.notes or ""
    )
    benchmark_id = db.add_benchmark(result)
    print(f"Added benchmark result: {benchmark_id}")
    print(f"  Engine: {args.engine}")
    print(f"  Benchmark: {args.name}")
    print(f"  Score: {args.score}")
    if parameters:
        print(f"  Parameters: {parameters}")


def _benchmark_leaderboard(db: ArcFusionDB, args: argparse.Namespace) -> None:
    """Show benchmark leaderboard."""
    if not args.name:
        _cli_error("--name required for leaderboard action")

    higher_better = not args.lower_better
    results = db.get_benchmark_leaderboard(
        args.name,
        higher_is_better=higher_better,
        limit=args.limit
    )

    if not results:
        print(f"No results for benchmark '{args.name}'")
        return

    direction = "↑" if higher_better else "↓"
    print(f"Leaderboard: {args.name} ({direction} higher is {'better' if higher_better else 'worse'})")
    print(SEPARATOR)
    for i, (engine, score) in enumerate(results, 1):
        print(f"  {i:2}. {engine.name:<30} {score:.4f}")


def _benchmark_show(db: ArcFusionDB, args: argparse.Namespace) -> None:
    """Show benchmark results for an engine."""
    if not args.engine:
        _cli_error("--engine required for show action")

    engine = db.get_engine_by_name(args.engine)
    if not engine:
        _cli_error(f"Engine '{args.engine}' not found")

    results = db.get_engine_benchmarks(engine.engine_id)
    if not results:
        print(f"No benchmark results for '{args.engine}'")
        return

    print(f"Benchmark results for: {args.engine}")
    print(SEPARATOR)
    for r in results:
        print(f"  {r.benchmark_name}: {r.score:.4f}")
        if r.parameters:
            print(f"    Params: {r.parameters}")
        if r.notes:
            print(f"    Notes: {r.notes}")


def _benchmark_compare(db: ArcFusionDB, args: argparse.Namespace) -> None:
    """Compare benchmark results between two engines."""
    if not args.engine or not args.engine2:
        _cli_error("--engine and --engine2 required for compare action")

    if args.engine.lower() == args.engine2.lower():
        _cli_error("Cannot compare an engine with itself")

    engine1 = db.get_engine_by_name(args.engine)
    engine2 = db.get_engine_by_name(args.engine2)

    if not engine1:
        _cli_error(f"Engine '{args.engine}' not found")
    if not engine2:
        _cli_error(f"Engine '{args.engine2}' not found")

    comparison = db.compare_engines([engine1.engine_id, engine2.engine_id])
    if not comparison:
        print("No benchmarks found for these engines")
        return

    print(f"Comparison: {args.engine} vs {args.engine2}")
    print(SEPARATOR)
    print(f"  {'Benchmark':<25} {args.engine:<15} {args.engine2:<15} {'Diff':>10}")
    print(SEPARATOR)
    for bench_name in sorted(comparison.keys()):
        scores = comparison[bench_name]
        score1 = scores.get(engine1.engine_id)
        score2 = scores.get(engine2.engine_id)
        s1_str = f"{score1:.4f}" if score1 is not None else "N/A"
        s2_str = f"{score2:.4f}" if score2 is not None else "N/A"
        if score1 is not None and score2 is not None:
            diff = score1 - score2
            diff_str = f"{diff:+.4f}"
        else:
            diff_str = ""
        print(f"  {bench_name:<25} {s1_str:<15} {s2_str:<15} {diff_str:>10}")


def _benchmark_list(db: ArcFusionDB, args: argparse.Namespace) -> None:
    """List all benchmark types."""
    benchmarks = db.list_benchmarks()
    if not benchmarks:
        print("No benchmarks recorded yet.")
        return

    print(f"Benchmark types ({len(benchmarks)}):")
    print(SEPARATOR)
    for b in benchmarks:
        print(f"  {b['benchmark_name']}")
        print(f"    Engines: {b['num_engines']}, Avg: {b['avg_score']:.4f}")
        print(f"    Range: {b['min_score']:.4f} - {b['max_score']:.4f}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Manage benchmark results."""
    with ArcFusionDB(args.db) as db:
        actions = {
            "add": _benchmark_add,
            "leaderboard": _benchmark_leaderboard,
            "show": _benchmark_show,
            "compare": _benchmark_compare,
            "list": _benchmark_list,
        }
        action_fn = actions.get(args.action)
        if action_fn:
            action_fn(db, args)


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate PyTorch code from a dreamed architecture."""
    with ArcFusionDB(args.db) as db:
        gen = CodeGenerator(db)
        kwargs = _build_dream_kwargs(args)

        # Generate
        try:
            result = gen.generate_from_dream(args.strategy, name=args.name, **kwargs)
        except ValueError as e:
            _cli_error(f"Code generation failed: {e}")

        # Validate
        valid, error = result.validate_syntax()
        if not valid:
            _cli_error(f"Generated code has syntax error: {error}")

        # Output
        if args.output:
            result.save(args.output)
            print(f"Generated {result.name} with {result.num_components} components")
            print(f"Saved to: {args.output}")
        else:
            print(result.code)

        print("\nComponents used:")
        for name in result.component_names:
            print(f"  - {name}")


def _cmd_validate_cloud(args: argparse.Namespace) -> None:
    """Run validation on cloud GPU via Modal."""
    with ArcFusionDB(args.db) as db:
        gen = CodeGenerator(db)
        kwargs = _build_dream_kwargs(args)

        # Step 1: Dream and generate code
        print(f"Dreaming architecture using '{args.strategy}' strategy...")
        try:
            generated = gen.generate_from_dream(args.strategy, name=args.name, **kwargs)
        except ValueError as e:
            _cli_error(f"Dream failed: {e}")

        print(f"Generated: {generated.name} with {generated.num_components} components")
        for name in generated.component_names:
            print(f"  - {name}")
        print()

        # Step 2: Configure cloud training
        cloud_config = CloudConfig(
            d_model=args.d_model,
            vocab_size=args.vocab_size,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            learning_rate=args.lr,
        )

        # Step 3: Run cloud training
        trainer = CloudTrainer(config=cloud_config)
        result = trainer.train(
            code=generated.code,
            model_name=generated.name,
            verbose=True,
        )

        # Step 4: Report results
        print(f"\n{SECTION_SEPARATOR}")
        print("Cloud Validation Summary")
        print(SEPARATOR)
        print(f"  Model: {generated.name}")
        print(f"  Success: {'Yes' if result.success else 'No'}")
        print(f"  Parameters: {result.num_parameters:,}")

        if result.error:
            print(f"  Error: {result.error}")

        if result.success:
            print(f"  Final Loss: {result.final_loss:.4f}")
            print(f"  Perplexity: {result.perplexity:.2f}")
            print(f"  Training Steps: {result.steps_completed}")
            print(f"  Training Time: {result.training_time_seconds:.1f}s")

            if result.loss_history:
                print(f"\nLoss History ({len(result.loss_history)} checkpoints):")
                for entry in result.loss_history[-5:]:  # Last 5
                    print(f"    Step {entry['step']}: {entry['loss']:.4f}")

        # Step 5: Store results if requested
        if args.store and result.success:
            benchmark = BenchmarkResult(
                engine_id=generated.name,
                benchmark_name="cloud_validation",
                score=result.perplexity,
                parameters={
                    'd_model': cloud_config.d_model,
                    'vocab_size': cloud_config.vocab_size,
                    'max_steps': cloud_config.max_steps,
                    'gpu': cloud_config.gpu_type,
                },
                notes="Cloud-validated dreamed architecture via Modal"
            )
            db.add_benchmark(benchmark)
            print("\nStored cloud validation result")


def cmd_web(args: argparse.Namespace) -> None:
    """Start the web UI server."""
    if not HAS_WEB:
        _cli_error("Web UI requires FastAPI. Install with: pip install 'arcfusion[web]'")

    print("Starting ArcFusion Web UI...")
    print(f"  Database: {args.db}")
    print(f"  URL: http://{args.host}:{args.port}")
    print("\nPress Ctrl+C to stop.\n")
    run_server(host=args.host, port=args.port, db_path=args.db)


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate a dreamed architecture by building, training, and benchmarking."""
    # Check for cloud mode
    if getattr(args, 'cloud', False):
        if not HAS_CLOUD:
            _cli_error("Cloud training requires Modal. Install with: pip install 'arcfusion[cloud]'")
        _cmd_validate_cloud(args)
        return

    if not HAS_VALIDATOR:
        _cli_error("Validator requires PyTorch. Install with: pip install torch")

    with ArcFusionDB(args.db) as db:
        gen = CodeGenerator(db)
        kwargs = _build_dream_kwargs(args)

        # Step 1: Dream and generate code
        print(f"Dreaming architecture using '{args.strategy}' strategy...")
        try:
            generated = gen.generate_from_dream(args.strategy, name=args.name, **kwargs)
        except ValueError as e:
            _cli_error(f"Dream failed: {e}")

        print(f"Generated: {generated.name} with {generated.num_components} components")
        for name in generated.component_names:
            print(f"  - {name}")
        print()

        # Step 2: Configure validation
        model_config = ModelConfig(
            d_model=args.d_model,
            vocab_size=args.vocab_size,
            max_seq_len=args.seq_len,
            n_heads=args.n_heads,
        )
        training_config = TrainingConfig(
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            device=args.device,
        )

        # Step 3: Run validation pipeline
        pipeline = ValidationPipeline(
            db=db,
            model_config=model_config,
            training_config=training_config,
        )

        result = pipeline.validate(generated, verbose=True)

        # Step 4: Report results
        print(f"\n{SECTION_SEPARATOR}")
        print("Validation Summary")
        print(SEPARATOR)
        print(f"  Model: {result.model_name}")
        print(f"  Success: {'Yes' if result.success else 'No'}")
        print(f"  Parameters: {result.num_parameters:,}")

        if result.build_error:
            print(f"  Build Error: {result.build_error}")
        if result.train_error:
            print(f"  Train Error: {result.train_error}")

        if result.success:
            print(f"  Final Loss: {result.final_loss:.4f}")
            print(f"  Perplexity: {result.perplexity:.2f}")
            print(f"  Training Steps: {result.training_steps}")
            print(f"  Training Time: {result.training_time_seconds:.1f}s")

            if result.benchmarks:
                print("\nBenchmarks:")
                for name, score in result.benchmarks.items():
                    print(f"    {name}: {score:.4f}")

        # Step 5: Store results if requested
        if args.store and result.success:
            # Create a dreamed engine entry to associate results with
            composer = EngineComposer(db)
            components, _ = composer.dream(args.strategy, **kwargs)

            # Store benchmark results
            for bench_name, score in result.benchmarks.items():
                benchmark = BenchmarkResult(
                    engine_id=generated.name,  # Use name as pseudo-ID
                    benchmark_name=bench_name,
                    score=score,
                    parameters={
                        'd_model': model_config.d_model,
                        'vocab_size': model_config.vocab_size,
                        'max_steps': training_config.max_steps,
                    },
                    notes="Auto-validated dreamed architecture"
                )
                db.add_benchmark(benchmark)

            print(f"\nStored {len(result.benchmarks)} benchmark results")


def cmd_export_results(args: argparse.Namespace) -> None:
    """Export training results from DB to JSON."""
    import json

    with ArcFusionDB(args.db) as db:
        # Optionally update vs_baseline_pct first
        if args.update_baseline:
            updated = db.update_vs_baseline_pct()
            print(f"Updated vs_baseline_pct for {updated} runs")

        results = db.export_results_json(baseline_model=args.baseline)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Exported {len(results['results'])} results to {args.output}")
        else:
            print(json.dumps(results, indent=2, default=str))


def cmd_insights(args: argparse.Namespace) -> None:
    """List, generate, or query training insights."""
    with ArcFusionDB(args.db) as db:
        if args.action == "list":
            _insights_list(db, args)
        elif args.action == "generate":
            _insights_generate(db, args)
        elif args.action == "show":
            _insights_show(db, args)
        elif args.action == "for-dreaming":
            _insights_for_dreaming(db, args)


def _insights_list(db: ArcFusionDB, args: argparse.Namespace) -> None:
    """List training insights."""
    insights = db.get_insights(
        category=args.category,
        tags=args.tags,
        min_confidence=args.min_confidence,
        limit=args.limit,
    )

    if not insights:
        print("No insights found. Run 'arcfusion insights generate' to create some.")
        return

    print(f"Training Insights ({len(insights)}):")
    print(SEPARATOR)

    for insight in insights:
        confidence_bar = "*" * int(insight.confidence * 5)
        print(f"\n[{insight.category}] {insight.title}")
        print(f"  Confidence: {insight.confidence:.0%} {confidence_bar}")
        if insight.description:
            # Truncate long descriptions
            desc = insight.description[:100] + "..." if len(insight.description) > 100 else insight.description
            print(f"  {desc}")
        if insight.tags:
            print(f"  Tags: {insight.tags}")
        if insight.source_comparison:
            print(f"  Source: {insight.source_comparison}")


def _insights_generate(db: ArcFusionDB, args: argparse.Namespace) -> None:
    """Generate new insights from training data."""
    print("Generating insights from training data...")
    print(SEPARATOR)

    new_insights = generate_training_insights(db)

    if not new_insights:
        print("No new insights generated.")
        print("This could mean:")
        print("  - No training runs in database")
        print("  - All possible insights already exist")
        print("  - Insufficient data for comparisons")
        return

    print(f"\nGenerated {len(new_insights)} new insights:")
    for insight in new_insights:
        print(f"\n  [{insight.category}] {insight.title}")
        if insight.description:
            print(f"    {insight.description[:80]}...")


def _insights_show(db: ArcFusionDB, args: argparse.Namespace) -> None:
    """Show detailed info for a specific insight."""
    import json

    if not args.insight_id:
        _cli_error("--id required for show action")

    insight = db.get_insight(args.insight_id)
    if not insight:
        _cli_error(f"Insight '{args.insight_id}' not found")

    print(f"Insight: {insight.title}")
    print(SEPARATOR)
    print(f"  ID: {insight.insight_id}")
    print(f"  Category: {insight.category}")
    print(f"  Confidence: {insight.confidence:.0%}")
    print(f"  Created: {insight.created_at}")
    if insight.description:
        print(f"\n  Description:")
        print(f"    {insight.description}")
    if insight.source_run_id:
        print(f"\n  Source run: {insight.source_run_id}")
    if insight.source_comparison:
        print(f"  Comparison: {insight.source_comparison}")
    if insight.tags:
        print(f"  Tags: {insight.tags}")
    if insight.evidence_json:
        print(f"\n  Evidence:")
        try:
            evidence = json.loads(insight.evidence_json)
            for k, v in evidence.items():
                print(f"    {k}: {v}")
        except json.JSONDecodeError:
            print(f"    {insight.evidence_json}")


def _insights_for_dreaming(db: ArcFusionDB, args: argparse.Namespace) -> None:
    """Show insights formatted for use in dream strategies."""
    insights_dict = db.get_insights_for_dreaming()

    if not insights_dict:
        print("No insights available for dreaming.")
        print("Run 'arcfusion insights generate' first.")
        return

    print("Insights for Dream Strategies")
    print(SECTION_SEPARATOR)

    for category, insights in insights_dict.items():
        print(f"\n{category.upper()}:")
        for insight in insights[:3]:  # Top 3 per category
            print(f"  - {insight['title']} ({insight['confidence']:.0%})")
            if insight.get('description'):
                print(f"      {insight['description'][:60]}...")

    print(f"\n{SEPARATOR}")
    print("Use 'arcfusion dream results_aware' to apply these insights automatically.")


def cmd_recipe(args: argparse.Namespace) -> None:
    """Get model recipe (code) from a training run to rebuild the model."""
    with ArcFusionDB(args.db) as db:
        # Find the run
        if args.run_id:
            run = db.get_training_run(args.run_id)
        elif args.model_name:
            runs = db.list_training_runs(model_name=args.model_name, success_only=True, limit=1)
            run = runs[0] if runs else None
        else:
            _cli_error("Either --run-id or --model is required")
            return

        if not run:
            _cli_error(f"No training run found")
            return

        if not run.model_code:
            _cli_error(f"No model code stored for run {run.run_id}")
            return

        if args.output:
            with open(args.output, "w") as f:
                f.write(run.model_code)
            print(f"Saved model code to {args.output}")
            print(f"  Model: {run.model_name}")
            print(f"  PPL: {run.perplexity:.1f}")
            print(f"  Code hash: {run.code_hash}")
        else:
            print(f"# Model: {run.model_name}")
            print(f"# PPL: {run.perplexity:.1f}")
            print(f"# Code hash: {run.code_hash}")
            print(f"# Run ID: {run.run_id}")
            print()
            print(run.model_code)


def cmd_leaderboard(args: argparse.Namespace) -> None:
    """Show training runs leaderboard with rankings and tiers."""
    with ArcFusionDB(args.db) as db:
        runs = db.list_training_runs(success_only=True, limit=args.limit)

        if not runs:
            print("No training runs found.")
            return

        # Get baseline for comparison
        baseline_runs = [r for r in runs if r.model_name == args.baseline]
        baseline_ppl = baseline_runs[0].perplexity if baseline_runs else None
        baseline_time = baseline_runs[0].time_seconds if baseline_runs else None

        # Sort by the requested metric
        if args.sort == "ppl":
            runs = sorted(runs, key=lambda r: r.perplexity)
        elif args.sort == "time":
            runs = sorted(runs, key=lambda r: r.time_seconds)
        elif args.sort == "efficiency":
            # Lower PPL and lower time = better efficiency
            runs = sorted(runs, key=lambda r: r.perplexity * r.time_seconds)

        # Classify tiers
        def get_tier(run):
            if baseline_ppl is None:
                return "N/A"
            ppl_ratio = run.perplexity / baseline_ppl
            time_ratio = run.time_seconds / baseline_time if baseline_time else 1

            if ppl_ratio < 0.9 and time_ratio < 0.5:
                return "S"  # Best quality AND fast
            elif ppl_ratio < 0.95:
                return "A"  # Better quality
            elif ppl_ratio < 1.05:
                return "B"  # Similar quality
            elif time_ratio < 0.5:
                return "C"  # Worse quality but fast
            else:
                return "D"  # Worse quality

        # Print header
        print(f"{'Rank':<5} {'Model':<30} {'PPL':>8} {'Time':>8} {'vs Base':>9} {'Tier':>5}")
        print("-" * 70)

        for i, run in enumerate(runs, 1):
            vs_base = ""
            if baseline_ppl:
                pct = ((run.perplexity - baseline_ppl) / baseline_ppl) * 100
                vs_base = f"{pct:+.1f}%"

            tier = get_tier(run)
            model_name = run.model_name.replace("Transformer_", "")[:28]

            print(f"{i:<5} {model_name:<30} {run.perplexity:>8.1f} {run.time_seconds:>7.0f}s {vs_base:>9} {tier:>5}")

        # Summary
        print("-" * 70)
        print(f"Total: {len(runs)} runs | Baseline: {args.baseline}")
        if baseline_ppl:
            print(f"Baseline PPL: {baseline_ppl:.1f} | Baseline Time: {baseline_time:.0f}s")

        # Tier legend
        print("\nTier Legend:")
        print("  S: Best quality + Fast (<90% PPL, <50% time)")
        print("  A: Better quality (<95% PPL)")
        print("  B: Similar quality (95-105% PPL)")
        print("  C: Fast but worse quality (>105% PPL, <50% time)")
        print("  D: Worse quality and slower")


def cmd_summary(args: argparse.Namespace) -> None:
    """Manage summaries - compressed knowledge storage for context preservation."""
    from .db import Summary

    with ArcFusionDB(args.db) as db:
        action = args.action

        if action == "add":
            # Add a new summary
            if not args.title or not args.content:
                print("ERROR: --title and --content are required for 'add' action")
                return

            summary = Summary(
                summary_type=args.type or "knowledge",
                title=args.title,
                content=args.content,
                context_json=args.context or "",
                tags=args.tags or "",
                source_ref=args.source or "",
            )
            summary_id = db.add_summary(summary)
            print(f"Created summary: {summary_id}")
            print(f"  Type: {summary.summary_type}")
            print(f"  Title: {summary.title}")
            if summary.tags:
                print(f"  Tags: {summary.tags}")

        elif action == "list":
            # List summaries with optional filters
            summaries = db.list_summaries(
                summary_type=args.type,
                tags=args.tags,
                search=args.search,
                limit=args.limit
            )
            if not summaries:
                print("No summaries found.")
                return

            print(f"{'ID':<14} {'Type':<12} {'Title':<40} {'Tags':<20}")
            print("-" * 90)
            for s in summaries:
                title = s.title[:38] + ".." if len(s.title) > 40 else s.title
                tags = s.tags[:18] + ".." if len(s.tags) > 20 else s.tags
                print(f"{s.summary_id:<14} {s.summary_type:<12} {title:<40} {tags:<20}")
            print(f"\nTotal: {len(summaries)} summaries")

        elif action == "show":
            # Show a specific summary
            if not args.summary_id:
                print("ERROR: --id is required for 'show' action")
                return

            summary = db.get_summary(args.summary_id)
            if not summary:
                print(f"Summary not found: {args.summary_id}")
                return

            print(f"Summary: {summary.summary_id}")
            print(f"Type: {summary.summary_type}")
            print(f"Title: {summary.title}")
            print(f"Created: {summary.created_at}")
            if summary.tags:
                print(f"Tags: {summary.tags}")
            if summary.source_ref:
                print(f"Source: {summary.source_ref}")
            print("\n--- Content ---")
            print(summary.content)
            if summary.context_json:
                print("\n--- Context (JSON) ---")
                print(summary.context_json)

        elif action == "delete":
            # Delete a summary
            if not args.summary_id:
                print("ERROR: --id is required for 'delete' action")
                return

            if db.delete_summary(args.summary_id):
                print(f"Deleted summary: {args.summary_id}")
            else:
                print(f"Summary not found: {args.summary_id}")

        elif action == "recipe-cards":
            # Auto-generate recipe cards from top training runs
            runs = db.list_training_runs(success_only=True, limit=args.limit or 10)

            if not runs:
                print("No training runs found to generate recipe cards.")
                return

            created = 0
            for run in runs:
                if not run.model_code:
                    continue

                # Check if summary already exists for this run
                existing = db.list_summaries(
                    summary_type="recipe",
                    search=run.run_id,
                    limit=1
                )
                if existing:
                    continue

                # Generate recipe card summary
                ppl_str = f"{run.perplexity:.1f}" if run.perplexity else "N/A"
                time_str = f"{run.time_seconds:.1f}s" if run.time_seconds else "N/A"

                content = f"""## {run.model_name}

**Performance**: {ppl_str} PPL @ {time_str}
**Run ID**: {run.run_id}
**Created**: {run.created_at}

### Model Code
```python
{run.model_code}
```

### Notes
{run.notes or 'No notes'}
"""
                summary = Summary(
                    summary_type="recipe",
                    title=f"Recipe: {run.model_name}",
                    content=content,
                    tags=f"model,{run.model_name}",
                    source_ref=run.run_id,
                )
                db.add_summary(summary)
                created += 1
                print(f"Created recipe card: {run.model_name}")

            print(f"\nGenerated {created} recipe cards from {len(runs)} training runs.")


def main() -> None:
    """
    ArcFusion CLI entry point.

    Provides commands for managing ML architecture components:
    - init: Seed the database with Transformer and modern architecture components
    - stats: Display database statistics
    - list: List components, engines, papers, or benchmarks
    - show: Display details about a component or engine
    - dream: Compose new architectures using various strategies
    - generate: Generate runnable PyTorch code from dreamed architectures
    - validate: Build, train, and benchmark dreamed architectures (requires PyTorch)
    - ingest: Import papers from arXiv
    - analyze: Deep LLM-powered component extraction (requires ANTHROPIC_API_KEY)
    - dedup: Find and merge duplicate components
    - web: Start interactive web UI (requires FastAPI)
    """
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
    dream_parser = subparsers.add_parser(
        "dream",
        help="Dream up new architecture",
        description="Compose a new architecture using one of four strategies: "
                    "greedy (pick best compatible components), random (weighted sampling), "
                    "mutate (modify existing engine), crossover (combine two engines)."
    )
    dream_parser.add_argument(
        "strategy",
        choices=["greedy", "random", "mutate", "crossover"],
        help="Composition strategy"
    )
    _add_dream_strategy_args(dream_parser)
    dream_parser.add_argument("--name", "-n", help="Name for the recipe (used with --save)")
    dream_parser.add_argument("--save", "-s", action="store_true", help="Save recipe to database")

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
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate PyTorch code from dreamed architecture",
        description="Dream an architecture and generate runnable PyTorch nn.Module code. "
                    "The generated code includes component classes, the main architecture class, "
                    "and example usage."
    )
    gen_parser.add_argument(
        "strategy",
        choices=["greedy", "random", "mutate", "crossover"],
        help="Composition strategy for dreaming"
    )
    gen_parser.add_argument("--name", "-n", default="DreamedArchitecture", help="Class name for generated architecture (default: DreamedArchitecture)")
    gen_parser.add_argument("--output", "-o", help="Output .py file path (prints to stdout if not specified)")
    _add_dream_strategy_args(gen_parser, random_steps=6)

    # config (component configurations management)
    config_parser = subparsers.add_parser(
        "config",
        help="Manage component configurations",
        description="Extract, list, and manage proven component configurations (sub-architectures)."
    )
    config_parser.add_argument(
        "action",
        choices=["list", "extract", "show"],
        help="Action: list (show all), extract (from engine), show (details)"
    )
    config_parser.add_argument("--engine", "-e", help="Engine name for extract action")
    config_parser.add_argument("--id", dest="config_id", help="Configuration ID for show action")
    config_parser.add_argument("--min-size", type=int, default=2, help="Min components in config (default: 2)")
    config_parser.add_argument("--max-size", type=int, help="Max components in config")
    config_parser.add_argument("--min-score", type=float, help="Min score for list filter")
    config_parser.add_argument("--validated", action="store_true", help="Only show validated configs")
    config_parser.add_argument("--save", dest="dry_run", action="store_false", default=True, help="Save extracted configs (default is dry-run)")

    # benchmark (benchmark results management)
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Manage benchmark results",
        description="Add, view, and compare benchmark results for engines."
    )
    bench_parser.add_argument(
        "action",
        choices=["add", "list", "leaderboard", "show", "compare"],
        help="Action: add (record result), list (all benchmarks), leaderboard (top engines), show (engine results), compare (two engines)"
    )
    bench_parser.add_argument("--engine", "-e", help="Engine name")
    bench_parser.add_argument("--engine2", help="Second engine for compare action")
    bench_parser.add_argument("--name", "-n", help="Benchmark name (e.g., 'perplexity', 'glue', 'mmlu')")
    bench_parser.add_argument("--score", "-s", type=float, help="Benchmark score")
    bench_parser.add_argument("--params", nargs="*", help="Parameters as key=value pairs (e.g., layers=6 d_model=512)")
    bench_parser.add_argument("--notes", help="Notes about the benchmark run")
    bench_parser.add_argument("--limit", type=int, default=20, help="Max results for leaderboard (default: 20)")
    bench_parser.add_argument("--lower-better", action="store_true", help="Lower score is better (e.g., perplexity)")

    # validate (auto-validation pipeline)
    val_parser = subparsers.add_parser(
        "validate",
        help="Build, train, and benchmark a dreamed architecture (requires PyTorch)",
        description="Dream an architecture, compile it to a runnable PyTorch model, "
                    "train on synthetic data, and measure performance. This validates "
                    "that dreamed architectures actually work."
    )
    val_parser.add_argument(
        "strategy",
        choices=["greedy", "random", "mutate", "crossover"],
        help="Composition strategy for dreaming"
    )
    val_parser.add_argument("--name", "-n", default="ValidatedModel", help="Name for the model")
    # Dream strategy args
    _add_dream_strategy_args(val_parser)
    # Model config
    val_parser.add_argument("--d-model", dest="d_model", type=int, default=128, help="Model dimension (default: 128)")
    val_parser.add_argument("--vocab-size", dest="vocab_size", type=int, default=1000, help="Vocabulary size (default: 1000)")
    val_parser.add_argument("--seq-len", dest="seq_len", type=int, default=64, help="Sequence length (default: 64)")
    val_parser.add_argument("--n-heads", dest="n_heads", type=int, default=4, help="Attention heads (default: 4)")
    # Training config
    val_parser.add_argument("--batch-size", dest="batch_size", type=int, default=8, help="Batch size (default: 8)")
    val_parser.add_argument("--max-steps", dest="max_steps", type=int, default=100, help="Training steps (default: 100)")
    val_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    val_parser.add_argument("--device", default="cpu", help="Device: cpu or cuda (default: cpu)")
    val_parser.add_argument("--store", action="store_true", help="Store benchmark results in database")
    val_parser.add_argument("--cloud", action="store_true", help="Run training on cloud GPU via Modal")

    # export-results (export training results from DB)
    export_parser = subparsers.add_parser(
        "export-results",
        help="Export training results from DB to JSON",
        description="Export all training results from the database to JSON format. "
                    "Can regenerate results files from DB. Optionally updates vs_baseline_pct."
    )
    export_parser.add_argument("--output", "-o", help="Output JSON file (prints to stdout if not specified)")
    export_parser.add_argument("--baseline", "-b", default="Transformer_MHA", help="Baseline model name (default: Transformer_MHA)")
    export_parser.add_argument("--update-baseline", action="store_true", help="Update vs_baseline_pct for all runs before export")

    # insights (training insights management)
    insights_parser = subparsers.add_parser(
        "insights",
        help="List, generate, and query training insights",
        description="Auto-generated insights from training runs help inform dream strategies. "
                    "Insights compare model performance against baselines and detect patterns."
    )
    insights_parser.add_argument(
        "action",
        choices=["list", "generate", "show", "for-dreaming"],
        help="Action: list (show all), generate (create from training data), show (details), for-dreaming (formatted for dream strategies)"
    )
    insights_parser.add_argument("--category", "-c", help="Filter by category (architecture, attention, efficiency, training)")
    insights_parser.add_argument("--tags", "-t", help="Filter by tags (comma-separated)")
    insights_parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum confidence (0-1)")
    insights_parser.add_argument("--limit", "-l", type=int, default=20, help="Max results (default: 20)")
    insights_parser.add_argument("--id", dest="insight_id", help="Insight ID for show action")

    # recipe (get model code from training run)
    recipe_parser = subparsers.add_parser(
        "recipe",
        help="Get model code (recipe) to rebuild any trained model",
        description="Retrieve the full PyTorch model code for any training run. "
                    "Use this to rebuild or modify any model from the leaderboard."
    )
    recipe_parser.add_argument("--run-id", "-r", dest="run_id", help="Training run ID")
    recipe_parser.add_argument("--model", "-m", dest="model_name", help="Model name (uses most recent run)")
    recipe_parser.add_argument("--output", "-o", help="Output file (prints to stdout if not specified)")

    # leaderboard (ranked training runs with tiers)
    leaderboard_parser = subparsers.add_parser(
        "leaderboard",
        help="Show ranked leaderboard of training runs with tier classifications",
        description="Display all training runs sorted by quality or efficiency, with tier classifications."
    )
    leaderboard_parser.add_argument("--sort", "-s", choices=["ppl", "time", "efficiency"], default="ppl",
                                    help="Sort by: ppl (quality), time, efficiency (default: ppl)")
    leaderboard_parser.add_argument("--baseline", "-b", default="Transformer_MHA",
                                    help="Baseline model for comparison (default: Transformer_MHA)")
    leaderboard_parser.add_argument("--limit", "-l", type=int, default=50, help="Max results (default: 50)")

    # summary (compressed knowledge storage)
    summary_parser = subparsers.add_parser(
        "summary",
        help="Manage summaries - compressed knowledge for context preservation",
        description="Store and retrieve compressed knowledge summaries. "
                    "Types: session (work summaries), recipe (model cards), experiment (results), knowledge (learnings)."
    )
    summary_parser.add_argument(
        "action",
        choices=["add", "list", "show", "delete", "recipe-cards"],
        help="Action: add, list, show, delete, or recipe-cards (auto-generate from training runs)"
    )
    summary_parser.add_argument("--type", "-t", choices=["session", "recipe", "experiment", "knowledge"],
                                help="Summary type filter or type for new summary")
    summary_parser.add_argument("--title", help="Summary title (required for add)")
    summary_parser.add_argument("--content", help="Summary content (required for add)")
    summary_parser.add_argument("--tags", help="Comma-separated tags")
    summary_parser.add_argument("--context", help="JSON context data")
    summary_parser.add_argument("--source", help="Source reference (run_id, paper_id, etc.)")
    summary_parser.add_argument("--search", "-s", help="Search in title/content")
    summary_parser.add_argument("--id", dest="summary_id", help="Summary ID for show/delete")
    summary_parser.add_argument("--limit", "-l", type=int, default=20, help="Max results (default: 20)")

    # web (web UI server)
    web_parser = subparsers.add_parser(
        "web",
        help="Start the web UI server (requires FastAPI)",
        description="Start an interactive web interface to browse components, engines, "
                    "view relationship graphs, and dream new architectures."
    )
    web_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    web_parser.add_argument("--port", "-p", type=int, default=8000, help="Port to listen on (default: 8000)")

    args = parser.parse_args()

    # Command dispatch table
    commands = {
        "init": cmd_init,
        "stats": cmd_stats,
        "list": cmd_list,
        "show": cmd_show,
        "dream": cmd_dream,
        "ingest": cmd_ingest,
        "analyze": cmd_analyze,
        "dedup": cmd_dedup,
        "generate": cmd_generate,
        "config": cmd_config,
        "benchmark": cmd_benchmark,
        "validate": cmd_validate,
        "export-results": cmd_export_results,
        "insights": cmd_insights,
        "recipe": cmd_recipe,
        "leaderboard": cmd_leaderboard,
        "summary": cmd_summary,
        "web": cmd_web,
    }

    cmd_fn = commands.get(args.command)
    if cmd_fn:
        cmd_fn(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
