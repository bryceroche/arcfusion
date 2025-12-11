"""
ArcFusion Database - SQLite storage for ML architecture components and engines.

Tables:
- components: Reusable building blocks (attention, FFN, embeddings, etc.)
- engines: Complete architectures (Transformer, BERT, GPT, etc.)
- engine_components: Links engines to their components
- component_relationships: Tracks component compatibility (C2C scores)
- processed_papers: Papers already analyzed (deduplication)
- benchmark_results: Performance tracking for engines
"""

import sqlite3
import json
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class Component:
    """A reusable ML component (attention head, FFN layer, etc.)"""
    name: str
    description: str
    interface_in: dict
    interface_out: dict
    code: str = ""
    usefulness_score: float = 0.0
    component_id: str = ""
    # New fields for provenance and computation
    source_paper_id: str = ""  # arXiv ID where component was introduced
    introduced_year: int = 0   # Year component was introduced
    hyperparameters: dict = field(default_factory=dict)  # e.g., {"num_heads": 8, "dropout": 0.1}
    # Math/computation characteristics
    time_complexity: str = ""   # e.g., "O(n^2)", "O(n)", "O(n*d)"
    space_complexity: str = ""  # e.g., "O(n^2)", "O(n*d)"
    flops_formula: str = ""     # e.g., "4*n*d^2 + 2*n^2*d"
    is_parallelizable: bool = True  # Can be computed in parallel
    is_causal: bool = False     # Enforces causal/autoregressive constraint
    math_operations: list = field(default_factory=list)  # ["matmul", "softmax", "layernorm", "gelu"]

    def __post_init__(self):
        if not self.component_id:
            content = f"{self.name}{json.dumps(self.interface_in, sort_keys=True)}{json.dumps(self.interface_out, sort_keys=True)}"
            self.component_id = hashlib.sha256(content.encode()).hexdigest()[:12]
        if self.hyperparameters is None:
            self.hyperparameters = {}
        if self.math_operations is None:
            self.math_operations = []


@dataclass
class Engine:
    """A complete ML architecture composed of components"""
    name: str
    description: str
    paper_url: str = ""
    engine_score: float = 0.0
    engine_id: str = ""
    component_ids: list = field(default_factory=list)

    def __post_init__(self):
        if not self.engine_id:
            self.engine_id = hashlib.sha256(self.name.encode()).hexdigest()[:12]


@dataclass
class ComponentRelationship:
    """Tracks how well two components work together"""
    component1_id: str
    component2_id: str
    engine_id: str
    c2c_score: float


@dataclass
class ProcessedPaper:
    """Track papers already processed to avoid duplicates"""
    arxiv_id: str
    title: str
    engine_id: str = ""
    status: str = "processed"  # processed, skipped, failed
    notes: str = ""
    processed_at: str = ""


@dataclass
class BenchmarkResult:
    """Track benchmark results for engines"""
    engine_id: str
    benchmark_name: str
    score: float
    parameters: dict = field(default_factory=dict)
    notes: str = ""
    benchmark_id: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.benchmark_id:
            content = f"{self.engine_id}{self.benchmark_name}{json.dumps(self.parameters, sort_keys=True)}"
            self.benchmark_id = hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class DreamedEngine:
    """Track composer-generated architectures"""
    strategy: str  # greedy, random, mutate, crossover
    component_ids: list
    estimated_score: float = 0.0
    parent_engine_ids: list = field(default_factory=list)  # For crossover/mutate
    validated: bool = False
    actual_score: float = 0.0
    notes: str = ""
    dream_id: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.dream_id:
            content = f"{self.strategy}{json.dumps(self.component_ids, sort_keys=True)}{self.estimated_score}"
            self.dream_id = hashlib.sha256(content.encode()).hexdigest()[:12]
        if self.parent_engine_ids is None:
            self.parent_engine_ids = []


class ArcFusionDB:
    """SQLite database for ML architecture components and engines"""

    def __init__(self, db_path: str = "arcfusion.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Initialize database tables"""
        self.conn.executescript("""
            -- Components: reusable building blocks
            CREATE TABLE IF NOT EXISTS components (
                component_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                interface_in TEXT,
                interface_out TEXT,
                code TEXT,
                usefulness_score REAL DEFAULT 0.0,
                -- Provenance
                source_paper_id TEXT,
                introduced_year INTEGER,
                -- Hyperparameters
                hyperparameters TEXT,
                -- Math/computation characteristics
                time_complexity TEXT,
                space_complexity TEXT,
                flops_formula TEXT,
                is_parallelizable BOOLEAN DEFAULT 1,
                is_causal BOOLEAN DEFAULT 0,
                math_operations TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Engines: complete architectures
            CREATE TABLE IF NOT EXISTS engines (
                engine_id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                paper_url TEXT,
                engine_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Engine-component links with position
            CREATE TABLE IF NOT EXISTS engine_components (
                engine_id TEXT,
                component_id TEXT,
                position INTEGER,
                PRIMARY KEY (engine_id, component_id, position),
                FOREIGN KEY (engine_id) REFERENCES engines(engine_id),
                FOREIGN KEY (component_id) REFERENCES components(component_id)
            );

            -- Component-to-component relationships (per engine context)
            CREATE TABLE IF NOT EXISTS component_relationships (
                component1_id TEXT,
                component2_id TEXT,
                engine_id TEXT,
                c2c_score REAL DEFAULT 0.0,
                PRIMARY KEY (component1_id, component2_id, engine_id),
                FOREIGN KEY (component1_id) REFERENCES components(component_id),
                FOREIGN KEY (component2_id) REFERENCES components(component_id),
                FOREIGN KEY (engine_id) REFERENCES engines(engine_id)
            );

            -- Aggregate component compatibility (precomputed across engines)
            CREATE TABLE IF NOT EXISTS component_compatibility (
                component1_id TEXT,
                component2_id TEXT,
                aggregate_score REAL DEFAULT 0.0,
                sample_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (component1_id, component2_id),
                FOREIGN KEY (component1_id) REFERENCES components(component_id),
                FOREIGN KEY (component2_id) REFERENCES components(component_id)
            );

            -- Processed papers tracking
            CREATE TABLE IF NOT EXISTS processed_papers (
                arxiv_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                engine_id TEXT,
                status TEXT DEFAULT 'processed',
                notes TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (engine_id) REFERENCES engines(engine_id)
            );

            -- Benchmark results
            CREATE TABLE IF NOT EXISTS benchmark_results (
                benchmark_id TEXT PRIMARY KEY,
                engine_id TEXT NOT NULL,
                benchmark_name TEXT NOT NULL,
                score REAL NOT NULL,
                parameters TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (engine_id) REFERENCES engines(engine_id)
            );

            -- Dreamed engines: track composer outputs
            CREATE TABLE IF NOT EXISTS dreamed_engines (
                dream_id TEXT PRIMARY KEY,
                strategy TEXT NOT NULL,
                parent_engine_ids TEXT,
                component_ids TEXT NOT NULL,
                estimated_score REAL,
                validated BOOLEAN DEFAULT 0,
                actual_score REAL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_comp_usefulness ON components(usefulness_score DESC);
            CREATE INDEX IF NOT EXISTS idx_comp_complexity ON components(time_complexity);
            CREATE INDEX IF NOT EXISTS idx_engine_score ON engines(engine_score DESC);
            CREATE INDEX IF NOT EXISTS idx_c2c_score ON component_relationships(c2c_score DESC);
            CREATE INDEX IF NOT EXISTS idx_compat_score ON component_compatibility(aggregate_score DESC);
            CREATE INDEX IF NOT EXISTS idx_paper_status ON processed_papers(status);
            CREATE INDEX IF NOT EXISTS idx_bench_engine ON benchmark_results(engine_id);
            CREATE INDEX IF NOT EXISTS idx_bench_name ON benchmark_results(benchmark_name);
            CREATE INDEX IF NOT EXISTS idx_bench_score ON benchmark_results(score DESC);
            CREATE INDEX IF NOT EXISTS idx_dream_strategy ON dreamed_engines(strategy);
            CREATE INDEX IF NOT EXISTS idx_dream_validated ON dreamed_engines(validated);
        """)
        self.conn.commit()

    # -------------------------------------------------------------------------
    # Component operations
    # -------------------------------------------------------------------------
    def add_component(self, comp: Component) -> str:
        """Add or update a component"""
        self.conn.execute("""
            INSERT OR REPLACE INTO components
            (component_id, name, description, interface_in, interface_out, code, usefulness_score,
             source_paper_id, introduced_year, hyperparameters,
             time_complexity, space_complexity, flops_formula,
             is_parallelizable, is_causal, math_operations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            comp.component_id,
            comp.name,
            comp.description,
            json.dumps(comp.interface_in),
            json.dumps(comp.interface_out),
            comp.code,
            comp.usefulness_score,
            comp.source_paper_id or None,
            comp.introduced_year or None,
            json.dumps(comp.hyperparameters) if comp.hyperparameters else None,
            comp.time_complexity or None,
            comp.space_complexity or None,
            comp.flops_formula or None,
            1 if comp.is_parallelizable else 0,
            1 if comp.is_causal else 0,
            json.dumps(comp.math_operations) if comp.math_operations else None,
        ))
        self.conn.commit()
        return comp.component_id

    def get_component(self, component_id: str) -> Optional[Component]:
        """Retrieve a component by ID"""
        row = self.conn.execute(
            "SELECT * FROM components WHERE component_id = ?", (component_id,)
        ).fetchone()
        if row:
            return self._row_to_component(row)
        return None

    def delete_component(self, component_id: str) -> bool:
        """Delete a component and its relationships. Returns True if deleted."""
        # Remove from relationships
        self.conn.execute(
            "DELETE FROM component_relationships WHERE component1_id = ? OR component2_id = ?",
            (component_id, component_id)
        )
        # Remove from engine_components
        self.conn.execute(
            "DELETE FROM engine_components WHERE component_id = ?",
            (component_id,)
        )
        # Remove component
        cursor = self.conn.execute(
            "DELETE FROM components WHERE component_id = ?",
            (component_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def _safe_json_loads(self, value: str, default):
        """Safely parse JSON, returning default on error."""
        if not value:
            return default
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default

    def _row_to_component(self, row) -> Component:
        """Convert a database row to a Component object"""
        keys = row.keys()
        return Component(
            component_id=row['component_id'],
            name=row['name'],
            description=row['description'],
            interface_in=self._safe_json_loads(row['interface_in'], {}),
            interface_out=self._safe_json_loads(row['interface_out'], {}),
            code=row['code'] or "",
            usefulness_score=row['usefulness_score'] or 0.0,
            source_paper_id=row['source_paper_id'] or "" if 'source_paper_id' in keys else "",
            introduced_year=row['introduced_year'] or 0 if 'introduced_year' in keys else 0,
            hyperparameters=self._safe_json_loads(row['hyperparameters'], {}) if 'hyperparameters' in keys else {},
            time_complexity=row['time_complexity'] or "" if 'time_complexity' in keys else "",
            space_complexity=row['space_complexity'] or "" if 'space_complexity' in keys else "",
            flops_formula=row['flops_formula'] or "" if 'flops_formula' in keys else "",
            is_parallelizable=bool(row['is_parallelizable']) if 'is_parallelizable' in keys else True,
            is_causal=bool(row['is_causal']) if 'is_causal' in keys else False,
            math_operations=self._safe_json_loads(row['math_operations'], []) if 'math_operations' in keys else [],
        )

    def find_components(
        self,
        name_pattern: Optional[str] = None,
        min_score: Optional[float] = None,
        time_complexity: Optional[str] = None,
        is_parallelizable: Optional[bool] = None,
        is_causal: Optional[bool] = None
    ) -> list[Component]:
        """Search for components with optional filters"""
        query = "SELECT * FROM components WHERE 1=1"
        params = []
        if name_pattern:
            query += " AND name LIKE ?"
            params.append(f"%{name_pattern}%")
        if min_score is not None:
            query += " AND usefulness_score >= ?"
            params.append(min_score)
        if time_complexity:
            query += " AND time_complexity = ?"
            params.append(time_complexity)
        if is_parallelizable is not None:
            query += " AND is_parallelizable = ?"
            params.append(1 if is_parallelizable else 0)
        if is_causal is not None:
            query += " AND is_causal = ?"
            params.append(1 if is_causal else 0)
        query += " ORDER BY usefulness_score DESC"

        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_component(r) for r in rows]

    # -------------------------------------------------------------------------
    # Engine operations
    # -------------------------------------------------------------------------
    def add_engine(self, engine: Engine) -> str:
        """Add an engine and its component links"""
        self.conn.execute("""
            INSERT OR REPLACE INTO engines
            (engine_id, name, description, paper_url, engine_score)
            VALUES (?, ?, ?, ?, ?)
        """, (
            engine.engine_id,
            engine.name,
            engine.description,
            engine.paper_url,
            engine.engine_score
        ))

        for pos, comp_id in enumerate(engine.component_ids):
            self.conn.execute("""
                INSERT OR IGNORE INTO engine_components (engine_id, component_id, position)
                VALUES (?, ?, ?)
            """, (engine.engine_id, comp_id, pos))

        self.conn.commit()
        return engine.engine_id

    def get_engine(self, engine_id: str) -> Optional[Engine]:
        """Retrieve an engine with its components"""
        row = self.conn.execute(
            "SELECT * FROM engines WHERE engine_id = ?", (engine_id,)
        ).fetchone()
        if not row:
            return None

        comp_ids = [
            r['component_id'] for r in self.conn.execute(
                "SELECT component_id FROM engine_components WHERE engine_id = ? ORDER BY position",
                (engine_id,)
            ).fetchall()
        ]

        return Engine(
            engine_id=row['engine_id'],
            name=row['name'],
            description=row['description'],
            paper_url=row['paper_url'],
            engine_score=row['engine_score'],
            component_ids=comp_ids
        )

    def get_engine_by_name(self, name: str) -> Optional[Engine]:
        """Retrieve an engine by name"""
        row = self.conn.execute(
            "SELECT engine_id FROM engines WHERE name = ?", (name,)
        ).fetchone()
        if row:
            return self.get_engine(row['engine_id'])
        return None

    def list_engines(self) -> list[Engine]:
        """List all engines"""
        rows = self.conn.execute("SELECT engine_id FROM engines ORDER BY engine_score DESC").fetchall()
        return [self.get_engine(r['engine_id']) for r in rows]

    def delete_engine(self, engine_id: str) -> bool:
        """Delete an engine and its component links. Returns True if deleted."""
        # Remove from engine_components
        self.conn.execute(
            "DELETE FROM engine_components WHERE engine_id = ?",
            (engine_id,)
        )
        # Remove relationships associated with this engine
        self.conn.execute(
            "DELETE FROM component_relationships WHERE engine_id = ?",
            (engine_id,)
        )
        # Remove benchmarks
        self.conn.execute(
            "DELETE FROM benchmark_results WHERE engine_id = ?",
            (engine_id,)
        )
        # Remove the engine
        cursor = self.conn.execute(
            "DELETE FROM engines WHERE engine_id = ?",
            (engine_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # Relationship operations
    # -------------------------------------------------------------------------
    def add_relationship(self, rel: ComponentRelationship):
        """Record a component relationship"""
        self.conn.execute("""
            INSERT OR REPLACE INTO component_relationships
            (component1_id, component2_id, engine_id, c2c_score)
            VALUES (?, ?, ?, ?)
        """, (rel.component1_id, rel.component2_id, rel.engine_id, rel.c2c_score))
        self.conn.commit()

    def get_compatible_components(self, component_id: str, min_score: float = 0.5) -> list[tuple[str, float]]:
        """Find components that work well with the given component"""
        rows = self.conn.execute("""
            SELECT
                CASE WHEN component1_id = ? THEN component2_id ELSE component1_id END as partner_id,
                AVG(c2c_score) as avg_score
            FROM component_relationships
            WHERE component1_id = ? OR component2_id = ?
            GROUP BY partner_id
            HAVING avg_score >= ?
            ORDER BY avg_score DESC
        """, (component_id, component_id, component_id, min_score)).fetchall()
        return [(r['partner_id'], r['avg_score']) for r in rows]

    # -------------------------------------------------------------------------
    # Processed papers operations
    # -------------------------------------------------------------------------
    def _normalize_arxiv_id(self, arxiv_id: str) -> str:
        """Normalize arxiv ID to just the number"""
        if "arxiv.org" in arxiv_id:
            arxiv_id = arxiv_id.split("/")[-1]
        if arxiv_id and len(arxiv_id) > 2 and arxiv_id[-2] == "v" and arxiv_id[-1].isdigit():
            arxiv_id = arxiv_id[:-2]
        return arxiv_id.strip()

    def is_paper_processed(self, arxiv_id: str) -> bool:
        """Check if we've already processed this paper"""
        arxiv_id = self._normalize_arxiv_id(arxiv_id)
        row = self.conn.execute(
            "SELECT 1 FROM processed_papers WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        return row is not None

    def add_processed_paper(self, paper: ProcessedPaper) -> str:
        """Record that we've processed a paper"""
        arxiv_id = self._normalize_arxiv_id(paper.arxiv_id)
        self.conn.execute("""
            INSERT OR REPLACE INTO processed_papers
            (arxiv_id, title, engine_id, status, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (arxiv_id, paper.title, paper.engine_id or None, paper.status, paper.notes))
        self.conn.commit()
        return arxiv_id

    def get_processed_paper(self, arxiv_id: str) -> Optional[ProcessedPaper]:
        """Get info about a processed paper"""
        arxiv_id = self._normalize_arxiv_id(arxiv_id)
        row = self.conn.execute(
            "SELECT * FROM processed_papers WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        if row:
            return ProcessedPaper(
                arxiv_id=row['arxiv_id'],
                title=row['title'],
                engine_id=row['engine_id'] or "",
                status=row['status'],
                notes=row['notes'] or "",
                processed_at=row['processed_at']
            )
        return None

    def list_processed_papers(self, status: Optional[str] = None, limit: int = 100) -> list[ProcessedPaper]:
        """List processed papers"""
        query = "SELECT * FROM processed_papers"
        params = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY processed_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [
            ProcessedPaper(
                arxiv_id=r['arxiv_id'],
                title=r['title'],
                engine_id=r['engine_id'] or "",
                status=r['status'],
                notes=r['notes'] or "",
                processed_at=r['processed_at']
            ) for r in rows
        ]

    # -------------------------------------------------------------------------
    # Benchmark operations
    # -------------------------------------------------------------------------
    def add_benchmark(self, result: BenchmarkResult) -> str:
        """Record a benchmark result"""
        self.conn.execute("""
            INSERT OR REPLACE INTO benchmark_results
            (benchmark_id, engine_id, benchmark_name, score, parameters, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            result.benchmark_id,
            result.engine_id,
            result.benchmark_name,
            result.score,
            json.dumps(result.parameters),
            result.notes
        ))
        self.conn.commit()
        return result.benchmark_id

    def get_engine_benchmarks(self, engine_id: str) -> list[BenchmarkResult]:
        """Get all benchmark results for an engine"""
        rows = self.conn.execute(
            "SELECT * FROM benchmark_results WHERE engine_id = ? ORDER BY benchmark_name",
            (engine_id,)
        ).fetchall()
        return [
            BenchmarkResult(
                benchmark_id=r['benchmark_id'],
                engine_id=r['engine_id'],
                benchmark_name=r['benchmark_name'],
                score=r['score'],
                parameters=self._safe_json_loads(r['parameters'], {}),
                notes=r['notes'] or "",
                created_at=r['created_at']
            ) for r in rows
        ]

    def get_benchmark_leaderboard(self, benchmark_name: str, higher_is_better: bool = True, limit: int = 20) -> list[tuple[Engine, float]]:
        """Get top engines for a benchmark"""
        order = "DESC" if higher_is_better else "ASC"
        rows = self.conn.execute(f"""
            SELECT e.*, br.score
            FROM engines e
            JOIN benchmark_results br ON e.engine_id = br.engine_id
            WHERE br.benchmark_name = ?
            ORDER BY br.score {order}
            LIMIT ?
        """, (benchmark_name, limit)).fetchall()

        results = []
        for r in rows:
            engine = Engine(
                engine_id=r['engine_id'],
                name=r['name'],
                description=r['description'],
                paper_url=r['paper_url'],
                engine_score=r['engine_score']
            )
            results.append((engine, r['score']))
        return results

    def compare_engines(self, engine_ids: list[str]) -> dict[str, dict[str, float]]:
        """Compare engines across benchmarks"""
        if not engine_ids:
            return {}

        placeholders = ",".join("?" * len(engine_ids))
        rows = self.conn.execute(f"""
            SELECT engine_id, benchmark_name, score
            FROM benchmark_results
            WHERE engine_id IN ({placeholders})
            ORDER BY benchmark_name, engine_id
        """, engine_ids).fetchall()

        comparison = {}
        for r in rows:
            bench = r['benchmark_name']
            if bench not in comparison:
                comparison[bench] = {}
            comparison[bench][r['engine_id']] = r['score']
        return comparison

    def list_benchmarks(self) -> list[dict]:
        """List all benchmark types with stats"""
        rows = self.conn.execute("""
            SELECT
                benchmark_name,
                COUNT(*) as num_engines,
                AVG(score) as avg_score,
                MIN(score) as min_score,
                MAX(score) as max_score
            FROM benchmark_results
            GROUP BY benchmark_name
            ORDER BY benchmark_name
        """).fetchall()
        return [dict(r) for r in rows]

    # -------------------------------------------------------------------------
    # Dreamed engines operations
    # -------------------------------------------------------------------------
    def add_dreamed_engine(self, dream: DreamedEngine) -> str:
        """Record a dreamed architecture"""
        self.conn.execute("""
            INSERT OR REPLACE INTO dreamed_engines
            (dream_id, strategy, parent_engine_ids, component_ids, estimated_score, validated, actual_score, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dream.dream_id,
            dream.strategy,
            json.dumps(dream.parent_engine_ids) if dream.parent_engine_ids else None,
            json.dumps(dream.component_ids),
            dream.estimated_score,
            1 if dream.validated else 0,
            dream.actual_score,
            dream.notes
        ))
        self.conn.commit()
        return dream.dream_id

    def get_dreamed_engine(self, dream_id: str) -> Optional[DreamedEngine]:
        """Retrieve a dreamed engine by ID"""
        row = self.conn.execute(
            "SELECT * FROM dreamed_engines WHERE dream_id = ?", (dream_id,)
        ).fetchone()
        if row:
            return DreamedEngine(
                dream_id=row['dream_id'],
                strategy=row['strategy'],
                parent_engine_ids=self._safe_json_loads(row['parent_engine_ids'], []),
                component_ids=self._safe_json_loads(row['component_ids'], []),
                estimated_score=row['estimated_score'] or 0.0,
                validated=bool(row['validated']),
                actual_score=row['actual_score'] or 0.0,
                notes=row['notes'] or "",
                created_at=row['created_at']
            )
        return None

    def list_dreamed_engines(self, strategy: Optional[str] = None, validated: Optional[bool] = None, limit: int = 100) -> list[DreamedEngine]:
        """List dreamed engines with optional filters"""
        query = "SELECT * FROM dreamed_engines WHERE 1=1"
        params = []
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if validated is not None:
            query += " AND validated = ?"
            params.append(1 if validated else 0)
        query += " ORDER BY estimated_score DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [
            DreamedEngine(
                dream_id=r['dream_id'],
                strategy=r['strategy'],
                parent_engine_ids=self._safe_json_loads(r['parent_engine_ids'], []),
                component_ids=self._safe_json_loads(r['component_ids'], []),
                estimated_score=r['estimated_score'] or 0.0,
                validated=bool(r['validated']),
                actual_score=r['actual_score'] or 0.0,
                notes=r['notes'] or "",
                created_at=r['created_at']
            ) for r in rows
        ]

    def validate_dreamed_engine(self, dream_id: str, actual_score: float, notes: str = "") -> bool:
        """Mark a dreamed engine as validated with actual score"""
        cursor = self.conn.execute("""
            UPDATE dreamed_engines
            SET validated = 1, actual_score = ?, notes = ?
            WHERE dream_id = ?
        """, (actual_score, notes, dream_id))
        self.conn.commit()
        return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # Component compatibility operations (aggregate scores)
    # -------------------------------------------------------------------------
    def update_compatibility_scores(self):
        """Recompute aggregate compatibility scores from relationships"""
        self.conn.execute("""
            INSERT OR REPLACE INTO component_compatibility (component1_id, component2_id, aggregate_score, sample_count)
            SELECT
                component1_id,
                component2_id,
                AVG(c2c_score) as aggregate_score,
                COUNT(*) as sample_count
            FROM component_relationships
            GROUP BY component1_id, component2_id
        """)
        self.conn.commit()

    def get_aggregate_compatibility(self, component_id: str, min_score: float = 0.5) -> list[tuple[str, float, int]]:
        """Get precomputed compatibility scores for a component"""
        rows = self.conn.execute("""
            SELECT
                CASE WHEN component1_id = ? THEN component2_id ELSE component1_id END as partner_id,
                aggregate_score,
                sample_count
            FROM component_compatibility
            WHERE (component1_id = ? OR component2_id = ?)
            AND aggregate_score >= ?
            ORDER BY aggregate_score DESC
        """, (component_id, component_id, component_id, min_score)).fetchall()
        return [(r['partner_id'], r['aggregate_score'], r['sample_count']) for r in rows]

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    def stats(self) -> dict:
        """Get database statistics"""
        return {
            'components': self.conn.execute("SELECT COUNT(*) FROM components").fetchone()[0],
            'engines': self.conn.execute("SELECT COUNT(*) FROM engines").fetchone()[0],
            'relationships': self.conn.execute("SELECT COUNT(*) FROM component_relationships").fetchone()[0],
            'compatibility_pairs': self.conn.execute("SELECT COUNT(*) FROM component_compatibility").fetchone()[0],
            'processed_papers': self.conn.execute("SELECT COUNT(*) FROM processed_papers").fetchone()[0],
            'benchmarks': self.conn.execute("SELECT COUNT(*) FROM benchmark_results").fetchone()[0],
            'benchmark_types': self.conn.execute("SELECT COUNT(DISTINCT benchmark_name) FROM benchmark_results").fetchone()[0],
            'dreamed_engines': self.conn.execute("SELECT COUNT(*) FROM dreamed_engines").fetchone()[0],
            'validated_dreams': self.conn.execute("SELECT COUNT(*) FROM dreamed_engines WHERE validated = 1").fetchone()[0],
        }

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
