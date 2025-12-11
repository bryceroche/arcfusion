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
)
from .composer import EngineComposer
from .decomposer import PaperDecomposer
from .seeds import seed_transformers, seed_modern_architectures

__version__ = "0.1.0"
__all__ = [
    "ArcFusionDB",
    "Component",
    "Engine",
    "ComponentRelationship",
    "ProcessedPaper",
    "BenchmarkResult",
    "EngineComposer",
    "PaperDecomposer",
    "seed_transformers",
    "seed_modern_architectures",
]
