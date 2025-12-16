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
from datetime import datetime, timezone
from typing import Any, Optional
from pathlib import Path


def _to_json(obj: Any) -> Optional[str]:
    """Convert object to JSON string, returning None for empty/falsy values."""
    if not obj:
        return None
    return json.dumps(obj)


def _bool_to_int(value: bool) -> int:
    """Convert boolean to integer for SQLite storage."""
    return 1 if value else 0


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

    def __post_init__(self) -> None:
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

    def __post_init__(self) -> None:
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

    def __post_init__(self) -> None:
        if not self.benchmark_id:
            content = f"{self.engine_id}{self.benchmark_name}{json.dumps(self.parameters, sort_keys=True)}"
            self.benchmark_id = hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class DreamedEngine:
    """Track composer-generated architectures"""
    strategy: str  # greedy, random, mutate, crossover
    component_ids: list = field(default_factory=list)
    estimated_score: float = 0.0
    parent_engine_ids: list = field(default_factory=list)  # For crossover/mutate
    validated: bool = False
    actual_score: float = 0.0
    notes: str = ""
    dream_id: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.dream_id:
            content = f"{self.strategy}{json.dumps(self.component_ids, sort_keys=True)}{self.estimated_score}"
            self.dream_id = hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class ComponentConfiguration:
    """A proven sub-configuration of components that work well together."""
    name: str
    component_ids: list = field(default_factory=list)  # Ordered list of component IDs
    description: str = ""
    source_engine_id: str = ""  # Engine this config was extracted from
    config_score: float = 0.0  # How well this config performs
    usage_count: int = 0  # How often used in dreams
    validated: bool = False  # Has been tested/validated
    config_id: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.config_id:
            content = f"{self.name}{json.dumps(self.component_ids, sort_keys=True)}"
            self.config_id = hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class Recipe:
    """
    A dreamed architecture recipe from Composer to ML Agent.

    Contains ordered components plus assembly instructions that tell
    the ML Agent how to wire them together.
    """
    name: str
    component_ids: list = field(default_factory=list)  # Ordered list of component IDs
    assembly: dict = field(default_factory=dict)  # Assembly instructions
    strategy: str = ""  # Dream strategy used: greedy, random, crossover, mutate
    estimated_score: float = 0.0
    parent_engine_ids: list = field(default_factory=list)  # For crossover/mutate
    notes: str = ""
    recipe_id: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.recipe_id:
            content = f"{self.name}{json.dumps(self.component_ids, sort_keys=True)}{json.dumps(self.assembly, sort_keys=True)}"
            self.recipe_id = hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class RecipeAdjustment:
    """
    Tracks modifications made by ML Agent during training.

    When the ML Agent needs to deviate from the recipe to enable training,
    each adjustment is recorded here for reproducibility and composer feedback.
    """
    recipe_id: str
    adjustment_type: str  # e.g., "shape_fix", "layer_add", "param_change", "skip_component"
    original_value: str  # What the recipe specified
    adjusted_value: str  # What was actually used
    reason: str  # Why the adjustment was needed
    component_id: str = ""  # Which component was affected (if applicable)
    adjustment_id: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.adjustment_id:
            content = f"{self.recipe_id}{self.adjustment_type}{self.original_value}{self.adjusted_value}"
            self.adjustment_id = hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class DreamCandidate:
    """
    A candidate architecture from the dream pipeline with surrogate predictions.

    Stores all dreamed architectures (even those not trained) to:
    - Avoid re-dreaming identical architectures
    - Track surrogate model accuracy over time
    - Guide future dreaming with historical data
    """
    strategy: str  # greedy, random, crossover, mutate
    temperature: float = 0.0  # Temperature used for dreaming
    components_json: str = ""  # JSON list of component names
    # Architecture features for surrogate model
    n_layers: int = 4
    n_kv_heads: int = 8
    # Surrogate predictions
    predicted_ppl: float = 0.0
    predicted_time: float = 0.0
    # Training results (if trained)
    was_trained: bool = False
    training_run_id: str = ""  # Links to training_runs if trained
    actual_ppl: float = 0.0
    actual_time: float = 0.0
    # Metadata
    notes: str = ""
    candidate_id: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.candidate_id:
            content = f"{self.strategy}{self.temperature}{self.components_json}{self.n_layers}"
            self.candidate_id = hashlib.sha256(content.encode()).hexdigest()[:12]

    def get_components(self) -> list[str]:
        """Parse components_json into a list of component names."""
        if not self.components_json:
            return []
        try:
            return json.loads(self.components_json)
        except (json.JSONDecodeError, TypeError):
            return []

    @property
    def has_mamba(self) -> bool:
        """Check if architecture contains Mamba/SSM components."""
        comps = ' '.join(self.get_components()).lower()
        return any(x in comps for x in ['mamba', 'ssm', 's4', 'state space', 'selective'])

    @property
    def has_linear_attn(self) -> bool:
        """Check if architecture contains linear attention."""
        comps = ' '.join(self.get_components()).lower()
        return 'linear' in comps and 'attention' in comps

    @property
    def is_hybrid(self) -> bool:
        """Check if architecture is a hybrid (has both attention and SSM)."""
        comps = ' '.join(self.get_components()).lower()
        has_attn = 'attention' in comps
        has_ssm = any(x in comps for x in ['mamba', 'ssm', 's4'])
        return has_attn and has_ssm

    @property
    def arch_type(self) -> str:
        """Derive architecture type from components and n_kv_heads."""
        if self.has_mamba and not self.is_hybrid:
            return 'mamba'
        if self.has_linear_attn:
            return 'linear'
        if self.n_kv_heads == 1:
            return 'mqa'
        if self.n_kv_heads < 8:
            return 'gqa'
        return 'mha'


@dataclass
class TrainingRun:
    """
    Records a complete training run with hardware, config, and results.

    Used for tracking experiments and comparing architectures fairly.
    Links to benchmark_results for post-training evaluation.
    """
    model_name: str
    # Training configuration
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    vocab_size: int = 8000
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_steps: int = 5000
    seed: int = 42  # Random seed for reproducibility
    # Hardware
    gpu_type: str = "A10G"
    mixed_precision: bool = True
    # Results
    parameters: int = 0
    final_train_loss: float = 0.0
    eval_loss: float = 0.0
    perplexity: float = 0.0
    time_seconds: float = 0.0
    success: bool = False
    error: str = ""
    # Baseline comparison
    is_baseline: bool = False
    baseline_run_id: str = ""  # Links to the baseline run for comparison
    vs_baseline_pct: float = 0.0  # % difference from baseline (negative = better)
    # Reproducibility: exact model code used
    model_code: str = ""  # Full Python code for the model
    code_hash: str = ""  # SHA256 hash of model_code for matching
    # Metadata
    recipe_id: str = ""  # Links to recipe if dreamed
    engine_id: str = ""  # Links to engine if known architecture
    experiment_id: str = ""  # Links to experiment grouping
    notes: str = ""
    run_id: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.run_id:
            content = f"{self.model_name}{self.d_model}{self.n_layers}{self.max_steps}{self.gpu_type}{self.seed}{self.created_at}"
            self.run_id = hashlib.sha256(content.encode()).hexdigest()[:12]
        # Auto-compute code_hash if model_code provided but no hash
        if self.model_code and not self.code_hash:
            self.code_hash = hashlib.sha256(self.model_code.encode()).hexdigest()[:16]


@dataclass
class Experiment:
    """
    Groups related training runs into a named experiment.

    Use this to track hypothesis-driven experiments like:
    - "Hybrid attention patterns sweep"
    - "SSM vs Attention at different scales"
    - "GQA n_kv_heads ablation"
    """
    name: str
    description: str = ""
    hypothesis: str = ""  # What we're testing
    status: str = "in_progress"  # in_progress, completed, abandoned
    run_ids: list = field(default_factory=list)  # Training run IDs in this experiment
    tags: list = field(default_factory=list)  # For filtering: ["hybrid", "attention", "scaling"]
    experiment_id: str = ""
    created_at: str = ""
    completed_at: str = ""

    def __post_init__(self) -> None:
        if not self.experiment_id:
            content = f"{self.name}{self.description}{self.hypothesis}"
            self.experiment_id = hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class Finding:
    """
    A conclusion backed by experimental evidence.

    Records insights like "Mamba beats MHA by 20% at equal params"
    with links to the supporting training runs.
    """
    title: str  # e.g., "SSM outperforms attention on WikiText-2"
    description: str = ""  # Full explanation
    experiment_id: str = ""  # Which experiment produced this
    evidence_run_ids: list = field(default_factory=list)  # Supporting run IDs
    confidence: str = "medium"  # high, medium, low
    delta_vs_baseline: float = 0.0  # % change (negative = better)
    statistical_significance: float = 0.0  # p-value or confidence interval
    tags: list = field(default_factory=list)  # For filtering
    finding_id: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.finding_id:
            content = f"{self.title}{self.experiment_id}{json.dumps(self.evidence_run_ids)}"
            self.finding_id = hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class TrainingInsight:
    """
    Auto-generated insight from training runs.

    After each training run, insights are automatically generated by comparing
    to previous runs. These inform future dream strategies.

    Categories: 'architecture', 'attention', 'efficiency', 'training'
    """
    category: str  # architecture, attention, efficiency, training
    title: str  # Short summary: 'Attention at END > START'
    description: str = ""  # Full explanation
    source_run_id: str = ""  # Training run that led to this insight
    source_comparison: str = ""  # e.g., 'AttnFirst vs MambaHeavy'
    evidence_json: str = ""  # JSON with PPL comparisons, times, etc.
    confidence: float = 0.8  # 0.0-1.0
    tags: str = ""  # Comma-separated: 'mamba,hybrid,attention,position'
    insight_id: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.insight_id:
            content = f"{self.category}{self.title}{self.source_run_id}{self.source_comparison}"
            self.insight_id = hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class Summary:
    """
    Compressed knowledge storage for context preservation and retrieval.

    Types:
    - 'session': Work session summary (what was accomplished)
    - 'recipe': Architecture recipe card (what makes it tick)
    - 'experiment': Experiment findings summary
    - 'knowledge': General compressed knowledge

    Use cases:
    - Restore context after compaction
    - Feed compressed knowledge to dream engine
    - Query past findings efficiently
    """
    summary_type: str  # session, recipe, experiment, knowledge
    title: str  # Short descriptive title
    content: str  # The compressed summary text
    context_json: str = ""  # Optional structured data (run_ids, metrics, etc.)
    tags: str = ""  # Comma-separated for querying
    source_ref: str = ""  # What was summarized (run_id, model_name, etc.)
    summary_id: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.summary_id:
            ts = datetime.now(timezone.utc).timestamp()
            content = f"{self.summary_type}{self.title}{ts}"
            self.summary_id = hashlib.sha256(content.encode()).hexdigest()[:12]


class ArcFusionDB:
    """SQLite database for ML architecture components and engines"""

    def __init__(self, db_path: str = "arcfusion.db", check_same_thread: bool = True):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=check_same_thread)
        self.conn.row_factory = sqlite3.Row
        self._migrate()  # Run migrations BEFORE schema init (to add columns needed for new indexes)
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

            -- Component configurations: proven sub-architectures
            CREATE TABLE IF NOT EXISTS component_configurations (
                config_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                component_ids TEXT NOT NULL,
                source_engine_id TEXT,
                config_score REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                validated BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_engine_id) REFERENCES engines(engine_id)
            );

            -- Recipes: Composer output for ML Agent
            CREATE TABLE IF NOT EXISTS recipes (
                recipe_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                component_ids TEXT NOT NULL,
                assembly TEXT NOT NULL,
                strategy TEXT,
                estimated_score REAL DEFAULT 0.0,
                parent_engine_ids TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Recipe adjustments: ML Agent modifications during training
            CREATE TABLE IF NOT EXISTS recipe_adjustments (
                adjustment_id TEXT PRIMARY KEY,
                recipe_id TEXT NOT NULL,
                adjustment_type TEXT NOT NULL,
                original_value TEXT NOT NULL,
                adjusted_value TEXT NOT NULL,
                reason TEXT NOT NULL,
                component_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id),
                FOREIGN KEY (component_id) REFERENCES components(component_id)
            );

            -- Training runs: detailed experiment tracking
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                -- Training config
                d_model INTEGER NOT NULL,
                n_layers INTEGER NOT NULL,
                n_heads INTEGER NOT NULL,
                vocab_size INTEGER NOT NULL,
                batch_size INTEGER NOT NULL,
                learning_rate REAL NOT NULL,
                max_steps INTEGER NOT NULL,
                seed INTEGER DEFAULT 42,
                -- Hardware
                gpu_type TEXT NOT NULL,
                mixed_precision INTEGER DEFAULT 1,
                -- Results
                parameters INTEGER DEFAULT 0,
                final_train_loss REAL DEFAULT 0.0,
                eval_loss REAL DEFAULT 0.0,
                perplexity REAL DEFAULT 0.0,
                time_seconds REAL DEFAULT 0.0,
                success INTEGER DEFAULT 0,
                error TEXT,
                -- Baseline comparison
                is_baseline INTEGER DEFAULT 0,
                baseline_run_id TEXT,
                vs_baseline_pct REAL DEFAULT 0.0,
                -- Reproducibility: exact model code
                model_code TEXT,
                code_hash TEXT,
                -- Links
                recipe_id TEXT,
                engine_id TEXT,
                experiment_id TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id),
                FOREIGN KEY (engine_id) REFERENCES engines(engine_id),
                FOREIGN KEY (baseline_run_id) REFERENCES training_runs(run_id),
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            );

            -- Experiments: group related training runs
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                hypothesis TEXT,
                status TEXT DEFAULT 'in_progress',
                run_ids TEXT,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );

            -- Findings: conclusions backed by evidence
            CREATE TABLE IF NOT EXISTS findings (
                finding_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                experiment_id TEXT,
                evidence_run_ids TEXT,
                confidence TEXT DEFAULT 'medium',
                delta_vs_baseline REAL DEFAULT 0.0,
                statistical_significance REAL DEFAULT 0.0,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            );

            -- Training insights: auto-generated insights from training runs
            CREATE TABLE IF NOT EXISTS training_insights (
                insight_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                source_run_id TEXT,
                source_comparison TEXT,
                evidence_json TEXT,
                confidence REAL DEFAULT 0.8,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_run_id) REFERENCES training_runs(run_id)
            );

            -- Summaries: compressed knowledge storage
            CREATE TABLE IF NOT EXISTS summaries (
                summary_id TEXT PRIMARY KEY,
                summary_type TEXT NOT NULL,  -- session, recipe, experiment, knowledge
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                context_json TEXT,  -- Optional structured data
                tags TEXT,
                source_ref TEXT,  -- What was summarized
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Dream candidates: all architectures dreamed with surrogate scores
            CREATE TABLE IF NOT EXISTS dream_candidates (
                candidate_id TEXT PRIMARY KEY,
                strategy TEXT NOT NULL,  -- greedy, random, crossover, mutate
                temperature REAL DEFAULT 0.0,
                components_json TEXT,  -- JSON list of component names
                -- Architecture features (derived props like has_mamba computed from components_json)
                n_layers INTEGER DEFAULT 4,
                n_kv_heads INTEGER DEFAULT 8,
                -- Surrogate predictions
                predicted_ppl REAL DEFAULT 0.0,
                predicted_time REAL DEFAULT 0.0,
                -- Training results (if trained)
                was_trained INTEGER DEFAULT 0,
                training_run_id TEXT,  -- Links to training_runs if trained
                actual_ppl REAL DEFAULT 0.0,
                actual_time REAL DEFAULT 0.0,
                -- Metadata
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (training_run_id) REFERENCES training_runs(run_id)
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
            CREATE INDEX IF NOT EXISTS idx_config_score ON component_configurations(config_score DESC);
            CREATE INDEX IF NOT EXISTS idx_config_usage ON component_configurations(usage_count DESC);
            CREATE INDEX IF NOT EXISTS idx_config_validated ON component_configurations(validated);
            CREATE INDEX IF NOT EXISTS idx_recipe_strategy ON recipes(strategy);
            CREATE INDEX IF NOT EXISTS idx_recipe_score ON recipes(estimated_score DESC);
            CREATE INDEX IF NOT EXISTS idx_adjustment_recipe ON recipe_adjustments(recipe_id);
            CREATE INDEX IF NOT EXISTS idx_adjustment_type ON recipe_adjustments(adjustment_type);
            CREATE INDEX IF NOT EXISTS idx_run_model ON training_runs(model_name);
            CREATE INDEX IF NOT EXISTS idx_run_baseline ON training_runs(is_baseline);
            CREATE INDEX IF NOT EXISTS idx_run_success ON training_runs(success);
            CREATE INDEX IF NOT EXISTS idx_run_ppl ON training_runs(perplexity);
            CREATE INDEX IF NOT EXISTS idx_run_created ON training_runs(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_run_experiment ON training_runs(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_run_code_hash ON training_runs(code_hash);
            -- Experiments indexes
            CREATE INDEX IF NOT EXISTS idx_exp_status ON experiments(status);
            CREATE INDEX IF NOT EXISTS idx_exp_name ON experiments(name);
            CREATE INDEX IF NOT EXISTS idx_exp_created ON experiments(created_at DESC);
            -- Findings indexes
            CREATE INDEX IF NOT EXISTS idx_finding_exp ON findings(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_finding_confidence ON findings(confidence);
            CREATE INDEX IF NOT EXISTS idx_finding_created ON findings(created_at DESC);
            -- Training insights indexes
            CREATE INDEX IF NOT EXISTS idx_insight_category ON training_insights(category);
            CREATE INDEX IF NOT EXISTS idx_insight_run ON training_insights(source_run_id);
            CREATE INDEX IF NOT EXISTS idx_insight_confidence ON training_insights(confidence DESC);
            CREATE INDEX IF NOT EXISTS idx_insight_created ON training_insights(created_at DESC);
            -- Summary indexes
            CREATE INDEX IF NOT EXISTS idx_summary_type ON summaries(summary_type);
            CREATE INDEX IF NOT EXISTS idx_summary_source ON summaries(source_ref);
            CREATE INDEX IF NOT EXISTS idx_summary_created ON summaries(created_at DESC);
            -- Dream candidate indexes
            CREATE INDEX IF NOT EXISTS idx_dream_cand_strategy ON dream_candidates(strategy);
            CREATE INDEX IF NOT EXISTS idx_dream_cand_ppl ON dream_candidates(predicted_ppl);
            CREATE INDEX IF NOT EXISTS idx_dream_cand_trained ON dream_candidates(was_trained);
            CREATE INDEX IF NOT EXISTS idx_dream_cand_created ON dream_candidates(created_at DESC);
        """)
        self.conn.commit()

    def _migrate(self):
        """Run schema migrations for existing databases."""
        # Check if training_runs table exists (skip if fresh DB)
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='training_runs'"
        )
        if not cursor.fetchone():
            return  # Fresh database, nothing to migrate

        # Get existing columns in training_runs
        cursor = self.conn.execute("PRAGMA table_info(training_runs)")
        columns = {row[1] for row in cursor.fetchall()}

        # Migration: Add model_code, code_hash, experiment_id to training_runs
        migrations = [
            ("training_runs", "model_code", "TEXT"),
            ("training_runs", "code_hash", "TEXT"),
            ("training_runs", "experiment_id", "TEXT"),
        ]

        for table, column, col_type in migrations:
            if column not in columns:
                try:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                    self.conn.commit()
                except sqlite3.OperationalError:
                    pass  # Column already exists or other issue

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
            _to_json(comp.hyperparameters),
            comp.time_complexity or None,
            comp.space_complexity or None,
            comp.flops_formula or None,
            _bool_to_int(comp.is_parallelizable),
            _bool_to_int(comp.is_causal),
            _to_json(comp.math_operations),
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

    def _safe_json_loads(self, value: Optional[str], default: Any) -> Any:
        """Safely parse JSON, returning default on error."""
        if not value:
            return default
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default

    def _row_to_component(self, row: sqlite3.Row) -> Component:
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
            params.append(_bool_to_int(is_parallelizable))
        if is_causal is not None:
            query += " AND is_causal = ?"
            params.append(_bool_to_int(is_causal))
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

        rows = self.conn.execute(
            "SELECT component_id FROM engine_components WHERE engine_id = ? ORDER BY position",
            (engine_id,)
        ).fetchall()
        comp_ids = [r['component_id'] for r in rows]

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
            _to_json(dream.parent_engine_ids),
            json.dumps(dream.component_ids),
            dream.estimated_score,
            _bool_to_int(dream.validated),
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
            params.append(_bool_to_int(validated))
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
    # Component configuration operations
    # -------------------------------------------------------------------------
    def add_configuration(self, config: ComponentConfiguration) -> str:
        """Add a component configuration."""
        self.conn.execute("""
            INSERT OR REPLACE INTO component_configurations
            (config_id, name, description, component_ids, source_engine_id,
             config_score, usage_count, validated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            config.config_id,
            config.name,
            config.description,
            json.dumps(config.component_ids),
            config.source_engine_id or None,
            config.config_score,
            config.usage_count,
            _bool_to_int(config.validated)
        ))
        self.conn.commit()
        return config.config_id

    def get_configuration(self, config_id: str) -> Optional[ComponentConfiguration]:
        """Retrieve a configuration by ID."""
        row = self.conn.execute(
            "SELECT * FROM component_configurations WHERE config_id = ?", (config_id,)
        ).fetchone()
        if row:
            return self._row_to_configuration(row)
        return None

    def _row_to_configuration(self, row: sqlite3.Row) -> ComponentConfiguration:
        """Convert a database row to a ComponentConfiguration object."""
        return ComponentConfiguration(
            config_id=row['config_id'],
            name=row['name'],
            description=row['description'] or "",
            component_ids=self._safe_json_loads(row['component_ids'], []),
            source_engine_id=row['source_engine_id'] or "",
            config_score=row['config_score'] or 0.0,
            usage_count=row['usage_count'] or 0,
            validated=bool(row['validated']),
            created_at=row['created_at'] or ""
        )

    def find_configurations(
        self,
        min_score: Optional[float] = None,
        validated: Optional[bool] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None
    ) -> list[ComponentConfiguration]:
        """Find configurations matching criteria."""
        query = "SELECT * FROM component_configurations WHERE 1=1"
        params = []

        if min_score is not None:
            query += " AND config_score >= ?"
            params.append(min_score)
        if validated is not None:
            query += " AND validated = ?"
            params.append(_bool_to_int(validated))

        query += " ORDER BY config_score DESC, usage_count DESC"
        rows = self.conn.execute(query, params).fetchall()

        configs = [self._row_to_configuration(r) for r in rows]

        # Filter by size in Python (component_ids is JSON)
        if min_size is not None:
            configs = [c for c in configs if len(c.component_ids) >= min_size]
        if max_size is not None:
            configs = [c for c in configs if len(c.component_ids) <= max_size]

        return configs

    def increment_config_usage(self, config_id: str) -> bool:
        """Increment usage count for a configuration."""
        cursor = self.conn.execute("""
            UPDATE component_configurations
            SET usage_count = usage_count + 1
            WHERE config_id = ?
        """, (config_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def validate_configuration(self, config_id: str, score: float) -> bool:
        """Mark a configuration as validated with actual score."""
        cursor = self.conn.execute("""
            UPDATE component_configurations
            SET validated = 1, config_score = ?
            WHERE config_id = ?
        """, (score, config_id))
        self.conn.commit()
        return cursor.rowcount > 0

    def delete_configuration(self, config_id: str) -> bool:
        """Delete a configuration."""
        cursor = self.conn.execute(
            "DELETE FROM component_configurations WHERE config_id = ?",
            (config_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def find_configurations_containing(self, component_id: str) -> list[ComponentConfiguration]:
        """Find all configurations that contain a specific component."""
        # Since component_ids is stored as JSON, we use LIKE for matching
        rows = self.conn.execute("""
            SELECT * FROM component_configurations
            WHERE component_ids LIKE ?
            ORDER BY config_score DESC
        """, (f'%"{component_id}"%',)).fetchall()
        return [self._row_to_configuration(r) for r in rows]

    # -------------------------------------------------------------------------
    # Recipe operations
    # -------------------------------------------------------------------------
    def add_recipe(self, recipe: Recipe) -> str:
        """Add a recipe (Composer output for ML Agent)."""
        self.conn.execute("""
            INSERT OR REPLACE INTO recipes
            (recipe_id, name, component_ids, assembly, strategy, estimated_score, parent_engine_ids, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            recipe.recipe_id,
            recipe.name,
            json.dumps(recipe.component_ids),
            json.dumps(recipe.assembly),
            recipe.strategy or None,
            recipe.estimated_score,
            _to_json(recipe.parent_engine_ids),
            recipe.notes or None
        ))
        self.conn.commit()
        return recipe.recipe_id

    def get_recipe(self, recipe_id: str) -> Optional[Recipe]:
        """Retrieve a recipe by ID."""
        row = self.conn.execute(
            "SELECT * FROM recipes WHERE recipe_id = ?", (recipe_id,)
        ).fetchone()
        if row:
            return self._row_to_recipe(row)
        return None

    def _row_to_recipe(self, row: sqlite3.Row) -> Recipe:
        """Convert a database row to a Recipe object."""
        return Recipe(
            recipe_id=row['recipe_id'],
            name=row['name'],
            component_ids=self._safe_json_loads(row['component_ids'], []),
            assembly=self._safe_json_loads(row['assembly'], {}),
            strategy=row['strategy'] or "",
            estimated_score=row['estimated_score'] or 0.0,
            parent_engine_ids=self._safe_json_loads(row['parent_engine_ids'], []),
            notes=row['notes'] or "",
            created_at=row['created_at'] or ""
        )

    def list_recipes(
        self,
        strategy: Optional[str] = None,
        min_score: Optional[float] = None,
        limit: int = 100
    ) -> list[Recipe]:
        """List recipes with optional filters."""
        query = "SELECT * FROM recipes WHERE 1=1"
        params = []
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if min_score is not None:
            query += " AND estimated_score >= ?"
            params.append(min_score)
        query += " ORDER BY estimated_score DESC, created_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_recipe(r) for r in rows]

    def delete_recipe(self, recipe_id: str) -> bool:
        """Delete a recipe and its adjustments."""
        # First delete adjustments
        self.conn.execute(
            "DELETE FROM recipe_adjustments WHERE recipe_id = ?",
            (recipe_id,)
        )
        # Then delete recipe
        cursor = self.conn.execute(
            "DELETE FROM recipes WHERE recipe_id = ?",
            (recipe_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # Recipe adjustment operations
    # -------------------------------------------------------------------------
    def add_adjustment(self, adjustment: RecipeAdjustment) -> str:
        """Record an adjustment made by ML Agent during training."""
        self.conn.execute("""
            INSERT OR REPLACE INTO recipe_adjustments
            (adjustment_id, recipe_id, adjustment_type, original_value, adjusted_value, reason, component_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            adjustment.adjustment_id,
            adjustment.recipe_id,
            adjustment.adjustment_type,
            adjustment.original_value,
            adjustment.adjusted_value,
            adjustment.reason,
            adjustment.component_id or None
        ))
        self.conn.commit()
        return adjustment.adjustment_id

    def get_adjustments(self, recipe_id: str) -> list[RecipeAdjustment]:
        """Get all adjustments for a recipe."""
        rows = self.conn.execute(
            "SELECT * FROM recipe_adjustments WHERE recipe_id = ? ORDER BY created_at",
            (recipe_id,)
        ).fetchall()
        return [self._row_to_adjustment(r) for r in rows]

    def _row_to_adjustment(self, row: sqlite3.Row) -> RecipeAdjustment:
        """Convert a database row to a RecipeAdjustment object."""
        return RecipeAdjustment(
            adjustment_id=row['adjustment_id'],
            recipe_id=row['recipe_id'],
            adjustment_type=row['adjustment_type'],
            original_value=row['original_value'],
            adjusted_value=row['adjusted_value'],
            reason=row['reason'],
            component_id=row['component_id'] or "",
            created_at=row['created_at'] or ""
        )

    def list_adjustments_by_type(self, adjustment_type: str, limit: int = 100) -> list[RecipeAdjustment]:
        """List all adjustments of a specific type (useful for Composer learning)."""
        rows = self.conn.execute("""
            SELECT * FROM recipe_adjustments
            WHERE adjustment_type = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (adjustment_type, limit)).fetchall()
        return [self._row_to_adjustment(r) for r in rows]

    def get_adjustment_stats(self) -> dict:
        """Get statistics about adjustments (helps identify common issues)."""
        rows = self.conn.execute("""
            SELECT adjustment_type, COUNT(*) as count
            FROM recipe_adjustments
            GROUP BY adjustment_type
            ORDER BY count DESC
        """).fetchall()
        return {r['adjustment_type']: r['count'] for r in rows}

    # -------------------------------------------------------------------------
    # Training run operations
    # -------------------------------------------------------------------------
    def add_training_run(self, run: TrainingRun) -> str:
        """Record a training run."""
        self.conn.execute("""
            INSERT OR REPLACE INTO training_runs
            (run_id, model_name, d_model, n_layers, n_heads, vocab_size, batch_size,
             learning_rate, max_steps, seed, gpu_type, mixed_precision, parameters,
             final_train_loss, eval_loss, perplexity, time_seconds, success, error,
             is_baseline, baseline_run_id, vs_baseline_pct, model_code, code_hash,
             recipe_id, engine_id, experiment_id, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run.run_id, run.model_name, run.d_model, run.n_layers, run.n_heads,
            run.vocab_size, run.batch_size, run.learning_rate, run.max_steps,
            run.seed, run.gpu_type, 1 if run.mixed_precision else 0, run.parameters,
            run.final_train_loss, run.eval_loss, run.perplexity, run.time_seconds,
            1 if run.success else 0, run.error, 1 if run.is_baseline else 0,
            run.baseline_run_id or None, run.vs_baseline_pct,
            run.model_code or None, run.code_hash or None,
            run.recipe_id or None, run.engine_id or None, run.experiment_id or None, run.notes
        ))
        self.conn.commit()
        return run.run_id

    def get_training_run(self, run_id: str) -> TrainingRun | None:
        """Get a training run by ID."""
        row = self.conn.execute(
            "SELECT * FROM training_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return self._row_to_training_run(row) if row else None

    def _row_to_training_run(self, row) -> TrainingRun:
        """Convert DB row to TrainingRun dataclass."""
        keys = row.keys()
        return TrainingRun(
            run_id=row['run_id'],
            model_name=row['model_name'],
            d_model=row['d_model'],
            n_layers=row['n_layers'],
            n_heads=row['n_heads'],
            vocab_size=row['vocab_size'],
            batch_size=row['batch_size'],
            learning_rate=row['learning_rate'],
            max_steps=row['max_steps'],
            seed=row['seed'] if 'seed' in keys else 42,
            gpu_type=row['gpu_type'],
            mixed_precision=bool(row['mixed_precision']),
            parameters=row['parameters'],
            final_train_loss=row['final_train_loss'],
            eval_loss=row['eval_loss'],
            perplexity=row['perplexity'],
            time_seconds=row['time_seconds'],
            success=bool(row['success']),
            error=row['error'] or "",
            is_baseline=bool(row['is_baseline']),
            baseline_run_id=row['baseline_run_id'] or "",
            vs_baseline_pct=row['vs_baseline_pct'],
            # Reproducibility fields
            model_code=row['model_code'] or "" if 'model_code' in keys else "",
            code_hash=row['code_hash'] or "" if 'code_hash' in keys else "",
            # Links
            recipe_id=row['recipe_id'] or "",
            engine_id=row['engine_id'] or "",
            experiment_id=row['experiment_id'] or "" if 'experiment_id' in keys else "",
            notes=row['notes'] or "",
            created_at=row['created_at'] or ""
        )

    def list_training_runs(
        self,
        model_name: str | None = None,
        baseline_only: bool = False,
        success_only: bool = True,
        limit: int = 100
    ) -> list[TrainingRun]:
        """List training runs with optional filters."""
        query = "SELECT * FROM training_runs WHERE 1=1"
        params = []

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        if baseline_only:
            query += " AND is_baseline = 1"
        if success_only:
            query += " AND success = 1"

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_training_run(r) for r in rows]

    def get_latest_baseline(self) -> TrainingRun | None:
        """Get the most recent successful baseline run."""
        row = self.conn.execute("""
            SELECT * FROM training_runs
            WHERE is_baseline = 1 AND success = 1
            ORDER BY created_at DESC
            LIMIT 1
        """).fetchone()
        return self._row_to_training_run(row) if row else None

    def get_training_leaderboard(self, limit: int = 20) -> list[TrainingRun]:
        """Get top training runs by perplexity (lower is better)."""
        rows = self.conn.execute("""
            SELECT * FROM training_runs
            WHERE success = 1
            ORDER BY perplexity ASC
            LIMIT ?
        """, (limit,)).fetchall()
        return [self._row_to_training_run(r) for r in rows]

    def get_efficiency_leaderboard(self, limit: int = 20, time_penalty: float = 0.5,
                                    reference_time: float = 300.0) -> list[tuple[TrainingRun, float]]:
        """Get top training runs by efficiency score (balances PPL and training time).

        Efficiency = PPL * (time_seconds / reference_time) ^ time_penalty

        Args:
            limit: Number of results to return
            time_penalty: Exponent for time penalty (0.5 = sqrt, lower = less penalty)
            reference_time: Reference "good" training time in seconds

        Returns:
            List of (TrainingRun, efficiency_score) tuples, sorted by efficiency (lower is better)
        """
        rows = self.conn.execute("""
            SELECT * FROM training_runs
            WHERE success = 1 AND perplexity > 0 AND time_seconds > 0
        """).fetchall()

        results = []
        for row in rows:
            run = self._row_to_training_run(row)
            efficiency = run.perplexity * (run.time_seconds / reference_time) ** time_penalty
            results.append((run, efficiency))

        # Sort by efficiency (lower is better) and return top N
        results.sort(key=lambda x: x[1])
        return results[:limit]

    def compare_to_baseline(self, run: TrainingRun, baseline: TrainingRun | None = None) -> float:
        """Calculate percentage difference from baseline. Negative = better."""
        if baseline is None:
            baseline = self.get_latest_baseline()
        if baseline is None or baseline.perplexity == 0:
            return 0.0
        return ((run.perplexity - baseline.perplexity) / baseline.perplexity) * 100

    def get_baseline_stats(self, model_name: str = "Transformer_MHA", config_hash: str = "") -> dict:
        """
        Get statistics for baseline runs (mean, std, count).

        Args:
            model_name: The baseline model name (default: Transformer_MHA)
            config_hash: Optional config hash to match specific hyperparameters

        Returns:
            Dict with: mean_ppl, std_ppl, n_runs, runs (list of TrainingRun)
        """
        import math

        query = """
            SELECT * FROM training_runs
            WHERE model_name = ? AND is_baseline = 1 AND success = 1
        """
        params = [model_name]

        if config_hash:
            query += " AND notes LIKE ?"
            params.append(f"%config_hash={config_hash}%")

        query += " ORDER BY created_at DESC"
        rows = self.conn.execute(query, params).fetchall()

        runs = [self._row_to_training_run(r) for r in rows]

        if not runs:
            return {"mean_ppl": 0.0, "std_ppl": 0.0, "n_runs": 0, "runs": []}

        perplexities = [r.perplexity for r in runs]
        mean_ppl = sum(perplexities) / len(perplexities)

        if len(perplexities) > 1:
            variance = sum((p - mean_ppl) ** 2 for p in perplexities) / (len(perplexities) - 1)
            std_ppl = math.sqrt(variance)
        else:
            std_ppl = 0.0

        return {
            "mean_ppl": mean_ppl,
            "std_ppl": std_ppl,
            "n_runs": len(runs),
            "runs": runs,
        }

    def get_baseline_seeds_needed(self, target_runs: int = 3, model_name: str = "Transformer_MHA", config_hash: str = "") -> list[int]:
        """
        Get list of seeds that still need to be run for baseline.

        Args:
            target_runs: How many baseline runs we want (default: 3)
            model_name: The baseline model name
            config_hash: Config hash to match

        Returns:
            List of seeds that haven't been run yet
        """
        stats = self.get_baseline_stats(model_name, config_hash)
        existing_seeds = {r.seed for r in stats["runs"]}

        # Standard seeds to use for baselines
        all_seeds = [42, 123, 456, 789, 1337]

        needed = []
        for seed in all_seeds:
            if seed not in existing_seeds and len(needed) + stats["n_runs"] < target_runs:
                needed.append(seed)

        return needed

    def export_results_json(
        self,
        baseline_model: str = "Transformer_MHA",
        include_config: bool = True
    ) -> dict:
        """
        Export training results to JSON format (can regenerate results files).

        Groups runs by model, calculates vs_baseline_pct, includes baseline stats.

        Returns:
            Dict suitable for JSON serialization with config, baseline, and results.
        """
        # Get baseline stats
        baseline_stats = self.get_baseline_stats(baseline_model)
        baseline_ppl = baseline_stats["mean_ppl"]

        # Get all successful runs (no limit for export)
        runs = self.list_training_runs(success_only=True, limit=10000)

        # Group by model name, keeping best (lowest ppl) for each
        best_by_model: dict[str, TrainingRun] = {}
        for run in runs:
            if run.model_name not in best_by_model:
                best_by_model[run.model_name] = run
            elif run.perplexity < best_by_model[run.model_name].perplexity:
                best_by_model[run.model_name] = run

        # Get baseline time for efficiency calculation
        baseline_time = baseline_stats["runs"][0].time_seconds if baseline_stats["runs"] else 0

        # Build results dict
        results = {}
        for model_name, run in best_by_model.items():
            vs_baseline = 0.0
            quality_score = 1.0
            speed_score = 1.0
            efficiency_score = 1.0

            if baseline_ppl > 0:
                quality_score = baseline_ppl / run.perplexity  # >1 = better quality
                if model_name != baseline_model:
                    vs_baseline = ((run.perplexity - baseline_ppl) / baseline_ppl) * 100

            if baseline_time > 0:
                speed_score = baseline_time / run.time_seconds  # >1 = faster

            # Efficiency = quality * speed (rewards both low ppl AND fast training)
            efficiency_score = quality_score * speed_score

            results[model_name] = {
                "success": run.success,
                "model_name": run.model_name,
                "parameters": run.parameters,
                "eval_loss": run.eval_loss,
                "perplexity": run.perplexity,
                "time_seconds": run.time_seconds,
                "vs_baseline_pct": vs_baseline,
                "quality_score": round(quality_score, 3),
                "speed_score": round(speed_score, 3),
                "efficiency_score": round(efficiency_score, 3),
                "run_id": run.run_id,
                "created_at": run.created_at,
            }

        output = {
            "baseline_model": baseline_model,
            "baseline_stats": {
                "mean_ppl": baseline_stats["mean_ppl"],
                "std_ppl": baseline_stats["std_ppl"],
                "n_runs": baseline_stats["n_runs"],
            },
            "results": results,
        }

        # Optionally include config from a baseline run
        if include_config and baseline_stats["runs"]:
            br = baseline_stats["runs"][0]
            output["config"] = {
                "d_model": br.d_model,
                "n_layers": br.n_layers,
                "n_heads": br.n_heads,
                "vocab_size": br.vocab_size,
                "batch_size": br.batch_size,
                "learning_rate": br.learning_rate,
                "max_steps": br.max_steps,
                "gpu_type": br.gpu_type,
                "mixed_precision": br.mixed_precision,
            }

        return output

    def update_vs_baseline_pct(self, baseline_model: str = "Transformer_MHA") -> int:
        """
        Backfill vs_baseline_pct for all runs based on baseline stats.

        Returns:
            Number of runs updated.
        """
        baseline_stats = self.get_baseline_stats(baseline_model)
        baseline_ppl = baseline_stats["mean_ppl"]

        if baseline_ppl <= 0:
            return 0

        runs = self.list_training_runs(success_only=True, limit=10000)
        updated = 0

        for run in runs:
            if run.model_name == baseline_model:
                # Set baseline to 0.0 explicitly
                self.conn.execute(
                    "UPDATE training_runs SET vs_baseline_pct = 0.0 WHERE run_id = ?",
                    (run.run_id,)
                )
                continue

            vs_pct = ((run.perplexity - baseline_ppl) / baseline_ppl) * 100

            self.conn.execute(
                "UPDATE training_runs SET vs_baseline_pct = ? WHERE run_id = ?",
                (vs_pct, run.run_id)
            )
            updated += 1

        self.conn.commit()
        return updated

    def get_model_performance_stats(
        self,
        baseline_model: str = "Transformer_MHA"
    ) -> dict:
        """
        Get structured analysis of past training results for ML researcher workflow.

        Returns performance analysis with rankings by quality, speed, and efficiency
        to inform architecture design decisions.

        Returns:
            Dict with:
            - rankings: Models ranked by quality, speed, efficiency
            - insights: Key observations (best quality, best speed, etc.)
            - recommendations: Suggestions for new architecture designs
        """
        results = self.export_results_json(baseline_model=baseline_model, include_config=False)

        if not results.get("results"):
            return {
                "rankings": {},
                "insights": [],
                "recommendations": ["No training data available. Run baseline benchmarks first."],
            }

        models = results["results"]

        # Sort by different metrics
        by_quality = sorted(models.items(), key=lambda x: x[1]["quality_score"], reverse=True)
        by_speed = sorted(models.items(), key=lambda x: x[1]["speed_score"], reverse=True)
        by_efficiency = sorted(models.items(), key=lambda x: x[1]["efficiency_score"], reverse=True)

        # Extract attention types from model names (e.g., "Transformer_MHA" -> "MHA")
        def get_attention_type(name: str) -> str:
            if "_" in name:
                return name.split("_")[-1]
            return name

        # Build insights
        insights = []
        if by_quality:
            best_q = by_quality[0]
            insights.append(
                "Best quality: {} ({:.3f}x baseline perplexity)".format(
                    best_q[0], best_q[1]["quality_score"]
                )
            )
        if by_speed:
            best_s = by_speed[0]
            insights.append(
                "Best speed: {} ({:.3f}x baseline training time)".format(
                    best_s[0], best_s[1]["speed_score"]
                )
            )
        if by_efficiency:
            best_e = by_efficiency[0]
            insights.append(
                "Best efficiency: {} (quality {:.3f} * speed {:.3f} = {:.3f})".format(
                    best_e[0], best_e[1]["quality_score"],
                    best_e[1]["speed_score"], best_e[1]["efficiency_score"]
                )
            )

        # Generate recommendations based on patterns
        recommendations = []

        # Check if there's a quality-speed tradeoff
        if by_quality and by_speed and by_quality[0][0] != by_speed[0][0]:
            q_best = get_attention_type(by_quality[0][0])
            s_best = get_attention_type(by_speed[0][0])
            recommendations.append(
                "Consider hybrid: {} for quality + {} for speed".format(q_best, s_best)
            )

        # Check if hybrid exists and how it performs
        hybrid_results = [m for m in models.items() if "Hybrid" in m[0]]
        if hybrid_results:
            h = hybrid_results[0]
            if h[1]["quality_score"] > 1.0 and h[1]["speed_score"] < 1.0:
                recommendations.append(
                    "Existing hybrid {} has good quality but slow - try fewer SSM layers".format(h[0])
                )

        # Suggest exploration based on gaps
        attention_types = {get_attention_type(m) for m in models.keys()}
        unexplored = {"Linear", "Retention", "RWKV"} - attention_types
        if unexplored:
            recommendations.append(
                "Unexplored attention types: {}".format(", ".join(sorted(unexplored)))
            )

        return {
            "rankings": {
                "by_quality": [(m, d["quality_score"]) for m, d in by_quality],
                "by_speed": [(m, d["speed_score"]) for m, d in by_speed],
                "by_efficiency": [(m, d["efficiency_score"]) for m, d in by_efficiency],
            },
            "insights": insights,
            "recommendations": recommendations,
            "raw_results": models,
        }

    # -------------------------------------------------------------------------
    # Experiment operations
    # -------------------------------------------------------------------------
    def add_experiment(self, exp: Experiment) -> str:
        """Create or update an experiment."""
        self.conn.execute("""
            INSERT OR REPLACE INTO experiments
            (experiment_id, name, description, hypothesis, status, run_ids, tags, created_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), ?)
        """, (
            exp.experiment_id,
            exp.name,
            exp.description,
            exp.hypothesis,
            exp.status,
            json.dumps(exp.run_ids) if exp.run_ids else None,
            json.dumps(exp.tags) if exp.tags else None,
            exp.created_at or None,
            exp.completed_at or None
        ))
        self.conn.commit()
        return exp.experiment_id

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Get an experiment by ID."""
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        return self._row_to_experiment(row) if row else None

    def _row_to_experiment(self, row) -> Experiment:
        """Convert DB row to Experiment dataclass."""
        return Experiment(
            experiment_id=row['experiment_id'],
            name=row['name'],
            description=row['description'] or "",
            hypothesis=row['hypothesis'] or "",
            status=row['status'] or "in_progress",
            run_ids=self._safe_json_loads(row['run_ids'], []),
            tags=self._safe_json_loads(row['tags'], []),
            created_at=row['created_at'] or "",
            completed_at=row['completed_at'] or ""
        )

    def list_experiments(
        self,
        status: str | None = None,
        tag: str | None = None,
        limit: int = 100
    ) -> list[Experiment]:
        """List experiments with optional filters."""
        query = "SELECT * FROM experiments WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if tag:
            query += " AND tags LIKE ?"
            params.append(f'%"{tag}"%')

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_experiment(r) for r in rows]

    def add_run_to_experiment(self, experiment_id: str, run_id: str) -> bool:
        """Add a training run to an experiment."""
        exp = self.get_experiment(experiment_id)
        if not exp:
            return False

        if run_id not in exp.run_ids:
            exp.run_ids.append(run_id)
            self.conn.execute(
                "UPDATE experiments SET run_ids = ? WHERE experiment_id = ?",
                (json.dumps(exp.run_ids), experiment_id)
            )
            self.conn.commit()

            # Also update the run's experiment_id
            self.conn.execute(
                "UPDATE training_runs SET experiment_id = ? WHERE run_id = ?",
                (experiment_id, run_id)
            )
            self.conn.commit()
        return True

    def complete_experiment(self, experiment_id: str) -> bool:
        """Mark an experiment as completed."""
        from datetime import datetime
        cursor = self.conn.execute("""
            UPDATE experiments
            SET status = 'completed', completed_at = ?
            WHERE experiment_id = ?
        """, (datetime.now().isoformat(), experiment_id))
        self.conn.commit()
        return cursor.rowcount > 0

    def get_experiment_runs(self, experiment_id: str) -> list[TrainingRun]:
        """Get all training runs in an experiment."""
        exp = self.get_experiment(experiment_id)
        if not exp or not exp.run_ids:
            return []
        return [r for rid in exp.run_ids if (r := self.get_training_run(rid))]

    # -------------------------------------------------------------------------
    # Finding operations
    # -------------------------------------------------------------------------
    def add_finding(self, finding: Finding) -> str:
        """Record a finding (conclusion backed by evidence)."""
        self.conn.execute("""
            INSERT OR REPLACE INTO findings
            (finding_id, title, description, experiment_id, evidence_run_ids,
             confidence, delta_vs_baseline, statistical_significance, tags, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
        """, (
            finding.finding_id,
            finding.title,
            finding.description,
            finding.experiment_id or None,
            json.dumps(finding.evidence_run_ids) if finding.evidence_run_ids else None,
            finding.confidence,
            finding.delta_vs_baseline,
            finding.statistical_significance,
            json.dumps(finding.tags) if finding.tags else None,
            finding.created_at or None
        ))
        self.conn.commit()
        return finding.finding_id

    def get_finding(self, finding_id: str) -> Finding | None:
        """Get a finding by ID."""
        row = self.conn.execute(
            "SELECT * FROM findings WHERE finding_id = ?", (finding_id,)
        ).fetchone()
        return self._row_to_finding(row) if row else None

    def _row_to_finding(self, row) -> Finding:
        """Convert DB row to Finding dataclass."""
        return Finding(
            finding_id=row['finding_id'],
            title=row['title'],
            description=row['description'] or "",
            experiment_id=row['experiment_id'] or "",
            evidence_run_ids=self._safe_json_loads(row['evidence_run_ids'], []),
            confidence=row['confidence'] or "medium",
            delta_vs_baseline=row['delta_vs_baseline'] or 0.0,
            statistical_significance=row['statistical_significance'] or 0.0,
            tags=self._safe_json_loads(row['tags'], []),
            created_at=row['created_at'] or ""
        )

    def list_findings(
        self,
        experiment_id: str | None = None,
        confidence: str | None = None,
        tag: str | None = None,
        limit: int = 100
    ) -> list[Finding]:
        """List findings with optional filters."""
        query = "SELECT * FROM findings WHERE 1=1"
        params = []

        if experiment_id:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        if confidence:
            query += " AND confidence = ?"
            params.append(confidence)
        if tag:
            query += " AND tags LIKE ?"
            params.append(f'%"{tag}"%')

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_finding(r) for r in rows]

    def get_findings_for_experiment(self, experiment_id: str) -> list[Finding]:
        """Get all findings associated with an experiment."""
        return self.list_findings(experiment_id=experiment_id)

    # -------------------------------------------------------------------------
    # Training insight operations
    # -------------------------------------------------------------------------
    def add_insight(self, insight: TrainingInsight) -> str:
        """Add a new training insight."""
        from datetime import datetime
        if not insight.insight_id:
            content = f"{insight.category}{insight.title}{insight.source_run_id}{insight.source_comparison}"
            insight.insight_id = hashlib.sha256(content.encode()).hexdigest()[:12]
        if not insight.created_at:
            insight.created_at = datetime.now().isoformat()

        self.conn.execute("""
            INSERT OR REPLACE INTO training_insights
            (insight_id, category, title, description, source_run_id, source_comparison,
             evidence_json, confidence, tags, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            insight.insight_id, insight.category, insight.title, insight.description,
            insight.source_run_id or None, insight.source_comparison or None,
            insight.evidence_json or None, insight.confidence,
            insight.tags or None, insight.created_at
        ))
        self.conn.commit()
        return insight.insight_id

    def get_insight(self, insight_id: str) -> TrainingInsight | None:
        """Get a training insight by ID."""
        row = self.conn.execute(
            "SELECT * FROM training_insights WHERE insight_id = ?", (insight_id,)
        ).fetchone()
        return self._row_to_insight(row) if row else None

    def _row_to_insight(self, row) -> TrainingInsight:
        """Convert DB row to TrainingInsight dataclass."""
        return TrainingInsight(
            insight_id=row['insight_id'],
            category=row['category'],
            title=row['title'],
            description=row['description'] or "",
            source_run_id=row['source_run_id'] or "",
            source_comparison=row['source_comparison'] or "",
            evidence_json=row['evidence_json'] or "",
            confidence=row['confidence'] or 0.8,
            tags=row['tags'] or "",
            created_at=row['created_at'] or ""
        )

    def get_insights(
        self,
        category: str | None = None,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
        limit: int = 100
    ) -> list[TrainingInsight]:
        """Get insights filtered by category and/or tags."""
        query = "SELECT * FROM training_insights WHERE confidence >= ?"
        params: list = [min_confidence]

        if category:
            query += " AND category = ?"
            params.append(category)

        if tags:
            for tag in tags:
                query += " AND tags LIKE ?"
                params.append(f"%{tag}%")

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_insight(r) for r in rows]

    def get_insights_for_dreaming(self) -> dict:
        """Get structured insights to inform dream strategies."""
        insights = self.get_insights(min_confidence=0.7)

        result: dict = {
            'architecture': [],
            'attention': [],
            'efficiency': [],
            'training': [],
            'recommendations': []
        }

        for i in insights:
            if i.category in result:
                result[i.category].append({
                    'title': i.title,
                    'description': i.description,
                    'confidence': i.confidence
                })

        # Generate recommendations from insights
        for i in insights:
            title_lower = i.title.lower()
            if 'best' in title_lower or 'better' in title_lower or '>' in i.title:
                result['recommendations'].append(i.description)

        return result

    def delete_insight(self, insight_id: str) -> bool:
        """Delete an insight."""
        cursor = self.conn.execute(
            "DELETE FROM training_insights WHERE insight_id = ?",
            (insight_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # Summary operations (compressed knowledge storage)
    # -------------------------------------------------------------------------
    def add_summary(self, summary: Summary) -> str:
        """Add a summary for compressed knowledge storage."""
        created_at = summary.created_at or datetime.now(timezone.utc).isoformat()

        self.conn.execute("""
            INSERT INTO summaries (
                summary_id, summary_type, title, content,
                context_json, tags, source_ref, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            summary.summary_id, summary.summary_type, summary.title, summary.content,
            summary.context_json, summary.tags, summary.source_ref, created_at
        ))
        self.conn.commit()
        return summary.summary_id

    def get_summary(self, summary_id: str) -> Summary | None:
        """Get a summary by ID."""
        row = self.conn.execute(
            "SELECT * FROM summaries WHERE summary_id = ?",
            (summary_id,)
        ).fetchone()
        if not row:
            return None
        return Summary(
            summary_id=row['summary_id'],
            summary_type=row['summary_type'],
            title=row['title'],
            content=row['content'],
            context_json=row['context_json'],
            tags=row['tags'],
            source_ref=row['source_ref'],
            created_at=row['created_at']
        )

    def list_summaries(
        self,
        summary_type: str | None = None,
        tags: str | None = None,
        search: str | None = None,
        limit: int = 50
    ) -> list[Summary]:
        """List summaries with optional filtering."""
        query = "SELECT * FROM summaries WHERE 1=1"
        params: list = []

        if summary_type:
            query += " AND summary_type = ?"
            params.append(summary_type)

        if tags:
            # Search for any of the comma-separated tags
            tag_conditions = []
            for tag in tags.split(','):
                tag = tag.strip()
                tag_conditions.append("tags LIKE ?")
                params.append(f"%{tag}%")
            if tag_conditions:
                query += f" AND ({' OR '.join(tag_conditions)})"

        if search:
            query += " AND (title LIKE ? OR content LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [
            Summary(
                summary_id=row['summary_id'],
                summary_type=row['summary_type'],
                title=row['title'],
                content=row['content'],
                context_json=row['context_json'],
                tags=row['tags'],
                source_ref=row['source_ref'],
                created_at=row['created_at']
            )
            for row in rows
        ]

    def delete_summary(self, summary_id: str) -> bool:
        """Delete a summary."""
        cursor = self.conn.execute(
            "DELETE FROM summaries WHERE summary_id = ?",
            (summary_id,)
        )
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
    # Dream candidate operations
    # -------------------------------------------------------------------------
    def add_dream_candidate(self, candidate: DreamCandidate) -> str:
        """Add a new dream candidate with surrogate predictions."""
        now = datetime.now(timezone.utc).isoformat()
        if not candidate.created_at:
            candidate.created_at = now

        self.conn.execute("""
            INSERT INTO dream_candidates (
                candidate_id, strategy, temperature, components_json,
                n_layers, n_kv_heads,
                predicted_ppl, predicted_time,
                was_trained, training_run_id, actual_ppl, actual_time,
                notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            candidate.candidate_id,
            candidate.strategy,
            candidate.temperature,
            candidate.components_json,
            candidate.n_layers,
            candidate.n_kv_heads,
            candidate.predicted_ppl,
            candidate.predicted_time,
            _bool_to_int(candidate.was_trained),
            candidate.training_run_id or None,
            candidate.actual_ppl,
            candidate.actual_time,
            candidate.notes,
            candidate.created_at
        ))
        self.conn.commit()
        return candidate.candidate_id

    def get_dream_candidate(self, candidate_id: str) -> DreamCandidate | None:
        """Get a dream candidate by ID."""
        row = self.conn.execute(
            "SELECT * FROM dream_candidates WHERE candidate_id = ?",
            (candidate_id,)
        ).fetchone()
        if not row:
            return None
        return DreamCandidate(
            strategy=row['strategy'],
            temperature=row['temperature'],
            components_json=row['components_json'] or "",
            n_layers=row['n_layers'],
            n_kv_heads=row['n_kv_heads'],
            predicted_ppl=row['predicted_ppl'],
            predicted_time=row['predicted_time'],
            was_trained=bool(row['was_trained']),
            training_run_id=row['training_run_id'] or "",
            actual_ppl=row['actual_ppl'],
            actual_time=row['actual_time'],
            notes=row['notes'] or "",
            candidate_id=row['candidate_id'],
            created_at=row['created_at'] or ""
        )

    def update_dream_candidate_training(
        self,
        candidate_id: str,
        training_run_id: str,
        actual_ppl: float,
        actual_time: float
    ) -> None:
        """Update a dream candidate with actual training results."""
        self.conn.execute("""
            UPDATE dream_candidates
            SET was_trained = 1, training_run_id = ?, actual_ppl = ?, actual_time = ?
            WHERE candidate_id = ?
        """, (training_run_id, actual_ppl, actual_time, candidate_id))
        self.conn.commit()

    def update_dream_candidate_predictions(
        self,
        candidate_id: str,
        predicted_ppl: float,
        predicted_time: float
    ) -> None:
        """Update a dream candidate's surrogate model predictions."""
        self.conn.execute("""
            UPDATE dream_candidates
            SET predicted_ppl = ?, predicted_time = ?
            WHERE candidate_id = ?
        """, (predicted_ppl, predicted_time, candidate_id))
        self.conn.commit()

    def list_dream_candidates(
        self,
        trained_only: bool = False,
        untrained_only: bool = False,
        limit: int = 100
    ) -> list[DreamCandidate]:
        """List dream candidates with optional filters."""
        query = "SELECT * FROM dream_candidates WHERE 1=1"
        params: list = []

        if trained_only:
            query += " AND was_trained = 1"
        if untrained_only:
            query += " AND was_trained = 0"

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [DreamCandidate(
            strategy=r['strategy'],
            temperature=r['temperature'],
            components_json=r['components_json'] or "",
            n_layers=r['n_layers'],
            n_kv_heads=r['n_kv_heads'],
            predicted_ppl=r['predicted_ppl'],
            predicted_time=r['predicted_time'],
            was_trained=bool(r['was_trained']),
            training_run_id=r['training_run_id'] or "",
            actual_ppl=r['actual_ppl'],
            actual_time=r['actual_time'],
            notes=r['notes'] or "",
            candidate_id=r['candidate_id'],
            created_at=r['created_at'] or ""
        ) for r in rows]

    def get_surrogate_accuracy(self) -> dict:
        """Calculate surrogate model accuracy on trained candidates."""
        rows = self.conn.execute("""
            SELECT predicted_ppl, actual_ppl, predicted_time, actual_time
            FROM dream_candidates
            WHERE was_trained = 1 AND actual_ppl > 0
        """).fetchall()

        if not rows:
            return {"n_samples": 0}

        ppl_errors = [abs(r['predicted_ppl'] - r['actual_ppl']) for r in rows]
        time_errors = [abs(r['predicted_time'] - r['actual_time']) for r in rows]

        return {
            "n_samples": len(rows),
            "ppl_mae": sum(ppl_errors) / len(ppl_errors),
            "time_mae": sum(time_errors) / len(time_errors),
            "ppl_max_error": max(ppl_errors),
            "time_max_error": max(time_errors),
        }

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    def stats(self) -> dict:
        """Get database statistics"""
        return {
            'components': self.conn.execute("SELECT COUNT(*) FROM components").fetchone()[0],
            'engines': self.conn.execute("SELECT COUNT(*) FROM engines").fetchone()[0],
            'configurations': self.conn.execute("SELECT COUNT(*) FROM component_configurations").fetchone()[0],
            'relationships': self.conn.execute("SELECT COUNT(*) FROM component_relationships").fetchone()[0],
            'compatibility_pairs': self.conn.execute("SELECT COUNT(*) FROM component_compatibility").fetchone()[0],
            'processed_papers': self.conn.execute("SELECT COUNT(*) FROM processed_papers").fetchone()[0],
            'benchmarks': self.conn.execute("SELECT COUNT(*) FROM benchmark_results").fetchone()[0],
            'benchmark_types': self.conn.execute("SELECT COUNT(DISTINCT benchmark_name) FROM benchmark_results").fetchone()[0],
            'dreamed_engines': self.conn.execute("SELECT COUNT(*) FROM dreamed_engines").fetchone()[0],
            'validated_dreams': self.conn.execute("SELECT COUNT(*) FROM dreamed_engines WHERE validated = 1").fetchone()[0],
            'recipes': self.conn.execute("SELECT COUNT(*) FROM recipes").fetchone()[0],
            'recipe_adjustments': self.conn.execute("SELECT COUNT(*) FROM recipe_adjustments").fetchone()[0],
            'training_runs': self.conn.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0],
            'successful_runs': self.conn.execute("SELECT COUNT(*) FROM training_runs WHERE success = 1").fetchone()[0],
            'baseline_runs': self.conn.execute("SELECT COUNT(*) FROM training_runs WHERE is_baseline = 1").fetchone()[0],
            'experiments': self.conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0],
            'experiments_completed': self.conn.execute("SELECT COUNT(*) FROM experiments WHERE status = 'completed'").fetchone()[0],
            'findings': self.conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0],
            'training_insights': self.conn.execute("SELECT COUNT(*) FROM training_insights").fetchone()[0],
            'dream_candidates': self.conn.execute("SELECT COUNT(*) FROM dream_candidates").fetchone()[0],
            'dream_candidates_trained': self.conn.execute("SELECT COUNT(*) FROM dream_candidates WHERE was_trained = 1").fetchone()[0],
        }

    def get_surrogate_accuracy_stats(self) -> dict:
        """Get accuracy statistics for surrogate model predictions.

        Compares predicted_ppl/predicted_time vs actual_ppl/actual_time
        for trained dream candidates.

        Returns:
            Dict with accuracy metrics, or empty dict if insufficient data.
        """
        rows = self.conn.execute("""
            SELECT predicted_ppl, actual_ppl, predicted_time, actual_time
            FROM dream_candidates
            WHERE was_trained = 1
              AND actual_ppl IS NOT NULL
              AND predicted_ppl IS NOT NULL
              AND predicted_ppl > 0
        """).fetchall()

        if len(rows) < 2:
            return {
                'n_samples': len(rows),
                'insufficient_data': True,
            }

        # Calculate metrics
        ppl_errors = []
        ppl_pct_errors = []
        time_errors = []
        time_pct_errors = []

        for pred_ppl, actual_ppl, pred_time, actual_time in rows:
            # PPL metrics
            ppl_errors.append(abs(pred_ppl - actual_ppl))
            ppl_pct_errors.append(abs(pred_ppl - actual_ppl) / actual_ppl * 100)

            # Time metrics (if available)
            if pred_time and actual_time and actual_time > 0:
                time_errors.append(abs(pred_time - actual_time))
                time_pct_errors.append(abs(pred_time - actual_time) / actual_time * 100)

        # Compute summary stats
        def mean(lst):
            return sum(lst) / len(lst) if lst else 0

        def correlation(pred, actual):
            """Pearson correlation coefficient."""
            if len(pred) < 2:
                return 0.0
            n = len(pred)
            mean_p = sum(pred) / n
            mean_a = sum(actual) / n
            num = sum((p - mean_p) * (a - mean_a) for p, a in zip(pred, actual))
            den_p = sum((p - mean_p) ** 2 for p in pred) ** 0.5
            den_a = sum((a - mean_a) ** 2 for a in actual) ** 0.5
            if den_p * den_a == 0:
                return 0.0
            return num / (den_p * den_a)

        pred_ppls = [r[0] for r in rows]
        actual_ppls = [r[1] for r in rows]
        pred_times = [r[2] for r in rows if r[2] and r[3]]
        actual_times = [r[3] for r in rows if r[2] and r[3]]

        return {
            'n_samples': len(rows),
            'insufficient_data': False,
            'ppl_mae': mean(ppl_errors),
            'ppl_mape': mean(ppl_pct_errors),  # Mean Absolute Percentage Error
            'ppl_correlation': correlation(pred_ppls, actual_ppls),
            'time_mae': mean(time_errors) if time_errors else None,
            'time_mape': mean(time_pct_errors) if time_pct_errors else None,
            'time_correlation': correlation(pred_times, actual_times) if len(pred_times) >= 2 else None,
        }

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def generate_training_insights(db: ArcFusionDB, new_run_id: str) -> list[TrainingInsight]:
    """
    Auto-generate insights by comparing new run to existing runs.

    Called after each training run to extract learnings. Creates insights
    that compare the new run against baseline and similar architectures.

    Args:
        db: Database connection
        new_run_id: ID of the just-completed training run

    Returns:
        List of TrainingInsight objects that were created and saved
    """
    new_run = db.get_training_run(new_run_id)
    if not new_run or not new_run.success:
        return []

    insights: list[TrainingInsight] = []
    all_runs = db.list_training_runs(success_only=True)

    def get_attn_type(name: str) -> str:
        """Extract attention type from model name."""
        if 'Mamba' in name and 'Hybrid' not in name and 'Heavy' not in name and 'First' not in name:
            return 'SSM'
        elif 'MHA' in name and 'Hybrid' not in name:
            return 'MHA'
        elif 'GQA' in name:
            return 'GQA'
        elif 'MQA' in name:
            return 'MQA'
        elif 'Hybrid' in name:
            return 'Hybrid'
        elif 'Heavy' in name:
            return 'MambaHeavy'
        elif 'First' in name:
            return 'AttnFirst'
        elif 'Linear' in name:
            return 'Linear'
        return 'Unknown'

    # Compare to baseline (MHA)
    baseline_runs = [r for r in all_runs if 'MHA' in r.model_name and 'Hybrid' not in r.model_name]
    if baseline_runs:
        baseline_ppl = sum(r.perplexity for r in baseline_runs) / len(baseline_runs)

        if new_run.perplexity < baseline_ppl * 0.95:  # 5%+ improvement
            quality_gain = (baseline_ppl - new_run.perplexity) / baseline_ppl * 100
            attn_type = get_attn_type(new_run.model_name)
            insights.append(TrainingInsight(
                source_run_id=new_run_id,
                source_comparison=f'{new_run.model_name} vs MHA baseline',
                category='architecture',
                title=f'{attn_type} beats baseline by {quality_gain:.1f}%',
                description=f'{new_run.model_name} achieved PPL {new_run.perplexity:.1f} vs baseline {baseline_ppl:.1f}. '
                           f'This {attn_type} architecture shows {quality_gain:.1f}% quality improvement.',
                evidence_json=json.dumps({
                    'new_ppl': new_run.perplexity,
                    'baseline_ppl': baseline_ppl,
                    'improvement_pct': quality_gain
                }),
                confidence=0.9 if quality_gain > 10 else 0.7,
                tags=f'{attn_type.lower()},quality,improvement'
            ))
        elif new_run.perplexity > baseline_ppl * 1.05:  # 5%+ worse
            quality_loss = (new_run.perplexity - baseline_ppl) / baseline_ppl * 100
            attn_type = get_attn_type(new_run.model_name)
            insights.append(TrainingInsight(
                source_run_id=new_run_id,
                source_comparison=f'{new_run.model_name} vs MHA baseline',
                category='architecture',
                title=f'{attn_type} underperforms baseline by {quality_loss:.1f}%',
                description=f'{new_run.model_name} achieved PPL {new_run.perplexity:.1f} vs baseline {baseline_ppl:.1f}. '
                           f'This architecture is {quality_loss:.1f}% worse than standard MHA.',
                evidence_json=json.dumps({
                    'new_ppl': new_run.perplexity,
                    'baseline_ppl': baseline_ppl,
                    'degradation_pct': quality_loss
                }),
                confidence=0.85,
                tags=f'{attn_type.lower()},quality,regression'
            ))

    # Compare similar architectures (same param count = fair comparison)
    similar_runs = [r for r in all_runs
                    if r.run_id != new_run_id
                    and abs(r.parameters - new_run.parameters) / max(new_run.parameters, 1) < 0.1]  # Within 10%

    for other in similar_runs:
        ppl_diff = abs(new_run.perplexity - other.perplexity)
        if ppl_diff > 5:  # Significant difference (5 PPL points)
            better = new_run if new_run.perplexity < other.perplexity else other
            worse = other if new_run.perplexity < other.perplexity else new_run
            better_type = get_attn_type(better.model_name)
            worse_type = get_attn_type(worse.model_name)

            # Only create insight if it's a meaningful comparison
            if better_type != worse_type:
                insights.append(TrainingInsight(
                    source_run_id=new_run_id,
                    source_comparison=f'{better.model_name} vs {worse.model_name}',
                    category='attention',
                    title=f'{better_type} > {worse_type} for quality',
                    description=f'{better.model_name} (PPL {better.perplexity:.1f}) outperforms '
                               f'{worse.model_name} (PPL {worse.perplexity:.1f}). '
                               f'Difference: {worse.perplexity - better.perplexity:.1f} PPL.',
                    evidence_json=json.dumps({
                        'better': {'model': better.model_name, 'ppl': better.perplexity},
                        'worse': {'model': worse.model_name, 'ppl': worse.perplexity}
                    }),
                    confidence=0.85,
                    tags=f'{better_type.lower()},{worse_type.lower()},comparison'
                ))

    # Check for speed vs quality tradeoffs
    if baseline_runs and new_run.time_seconds > 0:
        baseline_time = sum(r.time_seconds for r in baseline_runs) / len(baseline_runs)
        speed_ratio = new_run.time_seconds / baseline_time if baseline_time > 0 else 1.0
        quality_ratio = baseline_ppl / new_run.perplexity if new_run.perplexity > 0 else 1.0

        if speed_ratio > 2.0 and quality_ratio < 1.0:  # Slower AND worse
            insights.append(TrainingInsight(
                source_run_id=new_run_id,
                source_comparison=f'{new_run.model_name} efficiency',
                category='efficiency',
                title=f'{get_attn_type(new_run.model_name)} is slow AND lower quality',
                description=f'{new_run.model_name} took {speed_ratio:.1f}x longer than baseline '
                           f'but achieved {quality_ratio:.2f}x quality. Not recommended.',
                evidence_json=json.dumps({
                    'speed_ratio': speed_ratio,
                    'quality_ratio': quality_ratio,
                    'time_seconds': new_run.time_seconds,
                    'baseline_time': baseline_time
                }),
                confidence=0.9,
                tags='efficiency,speed,quality,tradeoff'
            ))
        elif speed_ratio < 0.5 and quality_ratio > 0.9:  # Much faster, acceptable quality
            insights.append(TrainingInsight(
                source_run_id=new_run_id,
                source_comparison=f'{new_run.model_name} efficiency',
                category='efficiency',
                title=f'{get_attn_type(new_run.model_name)} trades quality for speed',
                description=f'{new_run.model_name} trains {1/speed_ratio:.1f}x faster than baseline '
                           f'with {quality_ratio:.2f}x quality. Good for fast iteration.',
                evidence_json=json.dumps({
                    'speed_ratio': speed_ratio,
                    'quality_ratio': quality_ratio,
                    'speedup': 1/speed_ratio
                }),
                confidence=0.85,
                tags='efficiency,speed,fast'
            ))

    # Save all generated insights
    for insight in insights:
        db.add_insight(insight)

    return insights
