"""
Tests for the dedup module - duplicate detection and merging.
"""

import pytest
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from arcfusion.db import ArcFusionDB, Component
from arcfusion.dedup import (
    ComponentDeduplicator,
    DuplicateGroup,
    normalize_component_name,
    is_architecture_variant,
    extract_semantic_signature,
    calculate_similarity,
    find_duplicate_engines,
)
from arcfusion.seeds import seed_transformers


@pytest.fixture
def db():
    """Create a test database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    db = ArcFusionDB(db_path)
    yield db
    db.close()
    os.unlink(db_path)


class TestNormalizeComponentName:
    """Test name normalization."""

    def test_removes_parenthetical(self):
        assert normalize_component_name("RMSNorm (Root Mean Square)") == "rmsnorm"

    def test_lowercase(self):
        assert normalize_component_name("MultiHeadAttention") == "multiheadattention"

    def test_removes_special_chars(self):
        assert normalize_component_name("Position-wise Feed-Forward") == "positionwisefeedforward"

    def test_normalizes_plurals(self):
        assert normalize_component_name("Networks") == "network"
        assert normalize_component_name("Connections") == "connection"
        assert normalize_component_name("Embeddings") == "embedding"


class TestIsArchitectureVariant:
    """Test architecture variant detection."""

    def test_different_prefixes(self):
        assert is_architecture_variant("BERT Attention", "GPT Attention") is True

    def test_one_has_prefix(self):
        assert is_architecture_variant("Attention", "BERT Attention") is True

    def test_same_no_prefix(self):
        assert is_architecture_variant("Attention", "Multi-Head Attention") is False

    def test_sparse_variant(self):
        assert is_architecture_variant("Attention", "Sparse Attention") is True
        assert is_architecture_variant("Attention", "Block-Sparse Attention") is True


class TestExtractSemanticSignature:
    """Test semantic signature extraction."""

    def test_attention_terms(self):
        sig = extract_semantic_signature("Multi-Head Attention")
        assert "attention" in sig
        assert "multihead" in sig

    def test_normalization_terms(self):
        sig = extract_semantic_signature("RMSNorm Layer")
        assert "normalization" in sig
        assert "layer" in sig

    def test_feedforward_terms(self):
        sig = extract_semantic_signature("Position-wise FFN")
        assert "feedforward" in sig
        assert "positional" in sig

    def test_empty_name(self):
        sig = extract_semantic_signature("")
        assert sig == frozenset()


class TestCalculateSimilarity:
    """Test similarity calculation."""

    def test_identical_names(self):
        comp1 = Component(name="LayerNorm", description="", interface_in={}, interface_out={})
        comp2 = Component(name="LayerNorm", description="", interface_in={}, interface_out={})
        score, reason = calculate_similarity(comp1, comp2)
        assert score >= 0.6
        assert "same normalized name" in reason

    def test_similar_names(self):
        comp1 = Component(name="LayerNorm", description="", interface_in={}, interface_out={})
        comp2 = Component(name="Layer Normalization", description="", interface_in={}, interface_out={})
        score, reason = calculate_similarity(comp1, comp2)
        assert score >= 0.4

    def test_different_names(self):
        comp1 = Component(name="Attention", description="", interface_in={}, interface_out={})
        comp2 = Component(name="Dropout", description="", interface_in={}, interface_out={})
        score, reason = calculate_similarity(comp1, comp2)
        assert score < 0.5

    def test_same_source_paper(self):
        comp1 = Component(
            name="ModuleA", description="", interface_in={}, interface_out={},
            source_paper_id="1234.5678"
        )
        comp2 = Component(
            name="ModuleB", description="", interface_in={}, interface_out={},
            source_paper_id="1234.5678"
        )
        score, reason = calculate_similarity(comp1, comp2)
        assert "same source paper" in reason


class TestComponentDeduplicator:
    """Test duplicate finding and merging."""

    def test_find_no_duplicates(self, db):
        # Add distinct components
        db.add_component(Component(name="Attention", description="", interface_in={}, interface_out={}))
        db.add_component(Component(name="FeedForward", description="", interface_in={}, interface_out={}))
        db.add_component(Component(name="LayerNorm", description="", interface_in={}, interface_out={}))

        dedup = ComponentDeduplicator(db)
        groups = dedup.find_duplicates(threshold=0.5)
        assert len(groups) == 0

    def test_find_exact_duplicates(self, db):
        # Add components with same normalized name
        db.add_component(Component(
            name="LayerNorm", description="", interface_in={}, interface_out={},
            usefulness_score=0.9
        ))
        db.add_component(Component(
            name="Layer Norm", description="", interface_in={}, interface_out={},
            usefulness_score=0.8
        ))

        dedup = ComponentDeduplicator(db)
        groups = dedup.find_duplicates(threshold=0.5)
        assert len(groups) == 1
        assert groups[0].canonical.name == "LayerNorm"  # Higher score
        assert len(groups[0].duplicates) == 1

    def test_prefer_component_with_code(self, db):
        # Component with code should be canonical
        # Use slightly different names that normalize the same
        db.add_component(Component(
            name="Test Module", description="", interface_in={}, interface_out={},
            code="", usefulness_score=0.9
        ))
        db.add_component(Component(
            name="TestModule", description="", interface_in={}, interface_out={},
            code="def forward(self, x): return x", usefulness_score=0.8
        ))

        dedup = ComponentDeduplicator(db)
        groups = dedup.find_duplicates(threshold=0.5)
        assert len(groups) == 1
        # The one with code should be canonical despite lower score
        assert groups[0].canonical.code.strip() != ""

    def test_skip_architecture_variants(self, db):
        # These should NOT be merged
        db.add_component(Component(name="Attention", description="", interface_in={}, interface_out={}))
        db.add_component(Component(name="BERT Attention", description="", interface_in={}, interface_out={}))
        db.add_component(Component(name="Sparse Attention", description="", interface_in={}, interface_out={}))

        dedup = ComponentDeduplicator(db)
        groups = dedup.find_duplicates(threshold=0.5)
        # None should be grouped together
        assert len(groups) == 0

    def test_merge_group_dry_run(self, db):
        db.add_component(Component(name="Norm", description="", interface_in={}, interface_out={}))
        db.add_component(Component(name="Norm", description="", interface_in={}, interface_out={}))

        dedup = ComponentDeduplicator(db)
        groups = dedup.find_duplicates()

        if groups:
            result = dedup.merge_group(groups[0], dry_run=True)
            assert result['deleted'] == 0  # Dry run doesn't delete
            # Both components should still exist
            assert len(db.find_components("Norm")) == 2

    def test_merge_group_actual(self, db):
        c1 = Component(name="TestNorm", description="", interface_in={}, interface_out={}, usefulness_score=0.9)
        c2 = Component(name="TestNorm", description="", interface_in={}, interface_out={}, usefulness_score=0.8)
        db.add_component(c1)
        db.add_component(c2)

        dedup = ComponentDeduplicator(db)
        groups = dedup.find_duplicates()

        if groups:
            result = dedup.merge_group(groups[0], dry_run=False)
            assert result['deleted'] == 1
            # Only one should remain
            assert len(db.find_components("TestNorm")) == 1


class TestFindDuplicateEngines:
    """Test engine duplicate detection."""

    def test_find_duplicate_engines(self, db):
        from arcfusion.db import Engine

        db.add_engine(Engine(
            name="Transformer", description="", engine_score=0.9, component_ids=[]
        ))
        db.add_engine(Engine(
            name="Transformer (Vaswani)", description="", engine_score=0.8, component_ids=[]
        ))

        duplicates = find_duplicate_engines(db)
        assert len(duplicates) == 1
        # Both normalize to "transformer" so it's "same normalized name"
        assert "same normalized name" in duplicates[0][2] or "substring" in duplicates[0][2]

    def test_no_duplicate_engines(self, db):
        from arcfusion.db import Engine

        db.add_engine(Engine(name="BERT", description="", engine_score=0.9, component_ids=[]))
        db.add_engine(Engine(name="GPT", description="", engine_score=0.9, component_ids=[]))

        duplicates = find_duplicate_engines(db)
        assert len(duplicates) == 0
