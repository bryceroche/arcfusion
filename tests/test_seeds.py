"""
Tests for the seeds module - database seeding with pre-defined components and architectures.
"""

import pytest
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from arcfusion.db import ArcFusionDB
from arcfusion.seeds import (
    seed_transformers,
    seed_modern_architectures,
    TRANSFORMER_COMPONENTS,
    MODERN_COMPONENTS,
    ARCHITECTURES,
    COMPONENT_PAIRS,
)


@pytest.fixture
def db():
    """Create a fresh test database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    db = ArcFusionDB(db_path)
    yield db
    db.close()
    os.unlink(db_path)


class TestSeedData:
    """Test seed data definitions."""

    def test_transformer_components_defined(self):
        assert len(TRANSFORMER_COMPONENTS) >= 5
        names = [c.name for c in TRANSFORMER_COMPONENTS]
        assert "MultiHeadAttention" in names
        assert "FeedForward" in names
        assert "LayerNorm" in names

    def test_modern_components_defined(self):
        assert len(MODERN_COMPONENTS) >= 5
        names = [c.name for c in MODERN_COMPONENTS]
        assert "SelectiveSSM" in names
        assert "RMSNorm" in names
        assert "RotaryEmbedding" in names

    def test_architectures_defined(self):
        assert len(ARCHITECTURES) >= 5
        names = [a["name"] for a in ARCHITECTURES]
        assert "Transformer" in names
        assert "Mamba" in names
        assert "LLaMA" in names

    def test_component_pairs_defined(self):
        assert len(COMPONENT_PAIRS) >= 5
        # Each pair should have 3 elements: name1, name2, score
        for pair in COMPONENT_PAIRS:
            assert len(pair) == 3
            assert isinstance(pair[2], (int, float))
            assert 0 <= pair[2] <= 1


class TestSeedTransformers:
    """Test transformer seeding."""

    def test_seeds_components(self, db):
        name_to_id = seed_transformers(db, verbose=False)
        assert len(name_to_id) == len(TRANSFORMER_COMPONENTS)

    def test_components_in_db(self, db):
        seed_transformers(db, verbose=False)
        components = db.find_components()
        names = [c.name for c in components]
        assert "MultiHeadAttention" in names
        assert "Embedding" in names

    def test_returns_id_mapping(self, db):
        name_to_id = seed_transformers(db, verbose=False)
        assert "MultiHeadAttention" in name_to_id
        assert len(name_to_id["MultiHeadAttention"]) > 0  # UUID string

    def test_components_have_code(self, db):
        seed_transformers(db, verbose=False)
        components = db.find_components()
        # At least some components should have code
        components_with_code = [c for c in components if c.code and c.code.strip()]
        assert len(components_with_code) >= 3


class TestSeedModernArchitectures:
    """Test modern architecture seeding."""

    def test_seeds_after_transformers(self, db):
        seed_transformers(db, verbose=False)
        seed_modern_architectures(db, verbose=False)

        # Should have both transformer and modern components
        components = db.find_components()
        names = [c.name for c in components]
        assert "MultiHeadAttention" in names  # Transformer
        assert "SelectiveSSM" in names  # Modern

    def test_creates_engines(self, db):
        seed_transformers(db, verbose=False)
        seed_modern_architectures(db, verbose=False)

        engines = db.list_engines()
        names = [e.name for e in engines]
        assert "Transformer" in names
        assert "Mamba" in names
        assert "LLaMA" in names

    def test_engines_have_components(self, db):
        seed_transformers(db, verbose=False)
        seed_modern_architectures(db, verbose=False)

        transformer = db.get_engine_by_name("Transformer")
        assert transformer is not None
        assert len(transformer.component_ids) >= 3

    def test_creates_relationships(self, db):
        seed_transformers(db, verbose=False)
        seed_modern_architectures(db, verbose=False)

        stats = db.stats()
        assert stats["relationships"] > 0

    def test_idempotent_seeding(self, db):
        # Seeding twice should not create duplicates
        seed_transformers(db, verbose=False)
        seed_modern_architectures(db, verbose=False)

        count1 = len(db.list_engines())

        # Seed again
        seed_transformers(db, verbose=False)
        seed_modern_architectures(db, verbose=False)

        count2 = len(db.list_engines())
        assert count2 == count1  # No new engines created


class TestComponentQuality:
    """Test quality of seeded component data."""

    def test_components_have_interfaces(self, db):
        seed_transformers(db, verbose=False)
        components = db.find_components()

        for comp in components:
            assert comp.interface_in is not None
            assert comp.interface_out is not None

    def test_components_have_scores(self, db):
        seed_transformers(db, verbose=False)
        components = db.find_components()

        for comp in components:
            assert comp.usefulness_score > 0
            assert comp.usefulness_score <= 1.0

    def test_components_have_source_paper(self, db):
        seed_transformers(db, verbose=False)
        components = db.find_components()

        # Most should have source papers
        with_papers = [c for c in components if c.source_paper_id]
        assert len(with_papers) >= len(components) // 2

    def test_modern_components_have_complexity(self, db):
        seed_transformers(db, verbose=False)
        seed_modern_architectures(db, verbose=False)

        components = db.find_components()
        # Most should have time complexity
        with_complexity = [c for c in components if c.time_complexity]
        assert len(with_complexity) >= 5


class TestIntegration:
    """Integration tests for seeding."""

    def test_full_seed_stats(self, db):
        seed_transformers(db, verbose=False)
        seed_modern_architectures(db, verbose=False)

        stats = db.stats()
        assert stats["components"] >= 10
        assert stats["engines"] >= 5
        assert stats["relationships"] >= 10

    def test_seeded_db_supports_compose(self, db):
        from arcfusion.composer import EngineComposer

        seed_transformers(db, verbose=False)
        seed_modern_architectures(db, verbose=False)

        composer = EngineComposer(db)
        components, score = composer.dream("greedy")

        assert len(components) > 0
        assert score > 0
