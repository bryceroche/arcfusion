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

# Optional LLM-powered analyzer (requires anthropic package)
try:
    from .analyzer import PaperAnalyzer, AnalysisResult, AnalyzedComponent
    HAS_ANALYZER = True
except ImportError:
    HAS_ANALYZER = False
    PaperAnalyzer = None
    AnalysisResult = None
    AnalyzedComponent = None

__version__ = "0.1.0"
__all__ = [
    "ArcFusionDB",
    "Component",
    "Engine",
    "ComponentRelationship",
    "ProcessedPaper",
    "BenchmarkResult",
    "DreamedEngine",
    "EngineComposer",
    "PaperDecomposer",
    "ArxivFetcher",
    "ArxivPaper",
    "PaperAnalyzer",
    "AnalysisResult",
    "AnalyzedComponent",
    "HAS_ANALYZER",
    "seed_transformers",
    "seed_modern_architectures",
]
