"""
ArcFusion CLI - Command-line interface.
"""

import argparse
import sys
from .db import ArcFusionDB
from .composer import EngineComposer
from .seeds import seed_transformers, seed_modern_architectures


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

    components, score = composer.dream(args.strategy, **kwargs)

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
            print(f"  Interface in: {comp.interface_in}")
            print(f"  Interface out: {comp.interface_out}")

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
"""
    )
    parser.add_argument("--db", default="arcfusion.db", help="Database path")

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
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
