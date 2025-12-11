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

__version__ = "0.2.0"
__all__ = [
    # Database
    "ArcFusionDB",
    "Component",
    "Engine",
    "ComponentRelationship",
    "ProcessedPaper",
    "BenchmarkResult",
    "DreamedEngine",
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
    # Seeding
    "seed_transformers",
    "seed_modern_architectures",
]
