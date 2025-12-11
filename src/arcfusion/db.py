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

    def __post_init__(self):
        if not self.component_id:
            content = f"{self.name}{json.dumps(self.interface_in, sort_keys=True)}{json.dumps(self.interface_out, sort_keys=True)}"
            self.component_id = hashlib.sha256(content.encode()).hexdigest()[:12]


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
            CREATE TABLE IF NOT EXISTS components (
                component_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                interface_in TEXT,
                interface_out TEXT,
                code TEXT,
                usefulness_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS engines (
                engine_id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                paper_url TEXT,
                engine_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS engine_components (
                engine_id TEXT,
                component_id TEXT,
                position INTEGER,
                PRIMARY KEY (engine_id, component_id, position),
                FOREIGN KEY (engine_id) REFERENCES engines(engine_id),
                FOREIGN KEY (component_id) REFERENCES components(component_id)
            );

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

            CREATE TABLE IF NOT EXISTS processed_papers (
                arxiv_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                engine_id TEXT,
                status TEXT DEFAULT 'processed',
                notes TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (engine_id) REFERENCES engines(engine_id)
            );

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

            CREATE INDEX IF NOT EXISTS idx_comp_usefulness ON components(usefulness_score DESC);
            CREATE INDEX IF NOT EXISTS idx_engine_score ON engines(engine_score DESC);
            CREATE INDEX IF NOT EXISTS idx_c2c_score ON component_relationships(c2c_score DESC);
            CREATE INDEX IF NOT EXISTS idx_paper_status ON processed_papers(status);
            CREATE INDEX IF NOT EXISTS idx_bench_engine ON benchmark_results(engine_id);
            CREATE INDEX IF NOT EXISTS idx_bench_name ON benchmark_results(benchmark_name);
            CREATE INDEX IF NOT EXISTS idx_bench_score ON benchmark_results(score DESC);
        """)
        self.conn.commit()

    # -------------------------------------------------------------------------
    # Component operations
    # -------------------------------------------------------------------------
    def add_component(self, comp: Component) -> str:
        """Add or update a component"""
        self.conn.execute("""
            INSERT OR REPLACE INTO components
            (component_id, name, description, interface_in, interface_out, code, usefulness_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            comp.component_id,
            comp.name,
            comp.description,
            json.dumps(comp.interface_in),
            json.dumps(comp.interface_out),
            comp.code,
            comp.usefulness_score
        ))
        self.conn.commit()
        return comp.component_id

    def get_component(self, component_id: str) -> Optional[Component]:
        """Retrieve a component by ID"""
        row = self.conn.execute(
            "SELECT * FROM components WHERE component_id = ?", (component_id,)
        ).fetchone()
        if row:
            return Component(
                component_id=row['component_id'],
                name=row['name'],
                description=row['description'],
                interface_in=json.loads(row['interface_in']),
                interface_out=json.loads(row['interface_out']),
                code=row['code'],
                usefulness_score=row['usefulness_score']
            )
        return None

    def find_components(self, name_pattern: str = None, min_score: float = None) -> list[Component]:
        """Search for components"""
        query = "SELECT * FROM components WHERE 1=1"
        params = []
        if name_pattern:
            query += " AND name LIKE ?"
            params.append(f"%{name_pattern}%")
        if min_score is not None:
            query += " AND usefulness_score >= ?"
            params.append(min_score)
        query += " ORDER BY usefulness_score DESC"

        rows = self.conn.execute(query, params).fetchall()
        return [
            Component(
                component_id=r['component_id'],
                name=r['name'],
                description=r['description'],
                interface_in=json.loads(r['interface_in']),
                interface_out=json.loads(r['interface_out']),
                code=r['code'],
                usefulness_score=r['usefulness_score']
            ) for r in rows
        ]

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

    def list_processed_papers(self, status: str = None, limit: int = 100) -> list[ProcessedPaper]:
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
                parameters=json.loads(r['parameters']) if r['parameters'] else {},
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
    # Statistics
    # -------------------------------------------------------------------------
    def stats(self) -> dict:
        """Get database statistics"""
        return {
            'components': self.conn.execute("SELECT COUNT(*) FROM components").fetchone()[0],
            'engines': self.conn.execute("SELECT COUNT(*) FROM engines").fetchone()[0],
            'relationships': self.conn.execute("SELECT COUNT(*) FROM component_relationships").fetchone()[0],
            'processed_papers': self.conn.execute("SELECT COUNT(*) FROM processed_papers").fetchone()[0],
            'benchmarks': self.conn.execute("SELECT COUNT(*) FROM benchmark_results").fetchone()[0],
            'benchmark_types': self.conn.execute("SELECT COUNT(DISTINCT benchmark_name) FROM benchmark_results").fetchone()[0],
        }

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
