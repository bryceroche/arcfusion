"""
ArcFusion - ML Architecture Component Database & Composer

Decompose ML architectures into reusable components, track relationships,
and compose new configurations.
"""

from .db import (
    ArcFusionDB,
    Component,
    Engine,
    ComponentRelationship,
    ProcessedPaper,
    BenchmarkResult,
    DreamedEngine,
    ComponentConfiguration,
    Recipe,
    RecipeAdjustment,
)
from .composer import EngineComposer
from .decomposer import PaperDecomposer
from .fetcher import ArxivFetcher, ArxivPaper
from .seeds import seed_transformers, seed_modern_architectures
from .dedup import ComponentDeduplicator, DuplicateGroup
from .codegen import CodeGenerator, GeneratedCode

# Optional LLM-powered analyzer (requires anthropic package)
try:
    from .analyzer import PaperAnalyzer, AnalysisResult, AnalyzedComponent
    HAS_ANALYZER = True
except ImportError:
    HAS_ANALYZER = False
    PaperAnalyzer = None
    AnalysisResult = None
    AnalyzedComponent = None

# Optional validation pipeline (requires torch)
try:
    from .validator import (
        ValidationPipeline,
        ValidationResult,
        ModelBuilder,
        TrainingHarness,
        BenchmarkRunner,
        ModelConfig,
        TrainingConfig,
        HAS_TORCH,
    )
    # Only mark as available if torch is actually importable
    HAS_VALIDATOR = HAS_TORCH
except ImportError:
    HAS_VALIDATOR = False
    ValidationPipeline = None
    ValidationResult = None
    ModelBuilder = None
    TrainingHarness = None
    BenchmarkRunner = None
    ModelConfig = None
    TrainingConfig = None

# Optional ML Agent (requires torch)
try:
    from .ml_agent import MLAgent, ExecutionResult
    HAS_ML_AGENT = HAS_VALIDATOR  # ML Agent requires validator
except ImportError:
    HAS_ML_AGENT = False
    MLAgent = None
    ExecutionResult = None

__version__ = "0.2.0"
__all__ = [
    # Database
    "ArcFusionDB",
    "Component",
    "Engine",
    "ComponentRelationship",
    "ComponentConfiguration",
    "ProcessedPaper",
    "BenchmarkResult",
    "DreamedEngine",
    # Recipe system (Composer → ML Agent handoff)
    "Recipe",
    "RecipeAdjustment",
    # Composition
    "EngineComposer",
    "CodeGenerator",
    "GeneratedCode",
    # Deduplication
    "ComponentDeduplicator",
    "DuplicateGroup",
    # Paper processing
    "PaperDecomposer",
    "ArxivFetcher",
    "ArxivPaper",
    # LLM analysis (optional)
    "PaperAnalyzer",
    "AnalysisResult",
    "AnalyzedComponent",
    "HAS_ANALYZER",
    # Validation pipeline (optional)
    "ValidationPipeline",
    "ValidationResult",
    "ModelBuilder",
    "TrainingHarness",
    "BenchmarkRunner",
    "ModelConfig",
    "TrainingConfig",
    "HAS_VALIDATOR",
    # ML Agent (optional)
    "MLAgent",
    "ExecutionResult",
    "HAS_ML_AGENT",
    # Seeding
    "seed_transformers",
    "seed_modern_architectures",
]
