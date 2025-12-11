"""
Tests for the composer module - dream strategies and component composition.
"""

import pytest
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from arcfusion.db import ArcFusionDB, Component, Engine, ComponentRelationship
from arcfusion.composer import (
    EngineComposer,
    interfaces_compatible,
    get_component_category,
    normalize_shape,
    CATEGORY_ORDER,
)
from arcfusion.seeds import seed_transformers, seed_modern_architectures


@pytest.fixture
def db():
    """Create a seeded test database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    db = ArcFusionDB(db_path)
    seed_transformers(db, verbose=False)
    seed_modern_architectures(db, verbose=False)
    yield db
    db.close()
    os.unlink(db_path)


@pytest.fixture
def empty_db():
    """Create an empty test database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    db = ArcFusionDB(db_path)
    yield db
    db.close()
    os.unlink(db_path)


class TestNormalizeShape:
    """Test shape normalization for interface matching."""

    def test_normalize_basic(self):
        assert normalize_shape("[batch, seq_len, d_model]") == "[b,n,d]"

    def test_normalize_hidden(self):
        assert normalize_shape("[batch, seq_len, hidden]") == "[b,n,d]"

    def test_normalize_tgt_len(self):
        assert normalize_shape("[batch, tgt_len, d_model]") == "[b,n,d]"

    def test_normalize_empty(self):
        assert normalize_shape("") == ""
        assert normalize_shape(None) == ""


class TestInterfacesCompatible:
    """Test interface compatibility checking."""

    def test_exact_match(self):
        comp1 = Component(
            name="A", description="",
            interface_in={}, interface_out={"shape": "[batch, seq_len, d_model]"}
        )
        comp2 = Component(
            name="B", description="",
            interface_in={"shape": "[batch, seq_len, d_model]"}, interface_out={}
        )
        compatible, score = interfaces_compatible(comp1, comp2)
        assert compatible is True
        assert score == 1.0

    def test_normalized_match(self):
        comp1 = Component(
            name="A", description="",
            interface_in={}, interface_out={"shape": "[batch, seq_len, hidden]"}
        )
        comp2 = Component(
            name="B", description="",
            interface_in={"shape": "[batch, seq_len, d_model]"}, interface_out={}
        )
        compatible, score = interfaces_compatible(comp1, comp2)
        assert compatible is True
        assert score == 1.0

    def test_variable_shape(self):
        comp1 = Component(
            name="A", description="",
            interface_in={}, interface_out={"shape": "variable"}
        )
        comp2 = Component(
            name="B", description="",
            interface_in={"shape": "[batch, seq_len, d_model]"}, interface_out={}
        )
        compatible, score = interfaces_compatible(comp1, comp2)
        assert compatible is True
        assert score == 0.6

    def test_missing_shape(self):
        comp1 = Component(name="A", description="", interface_in={}, interface_out={})
        comp2 = Component(name="B", description="", interface_in={}, interface_out={})
        compatible, score = interfaces_compatible(comp1, comp2)
        assert compatible is True
        assert score == 0.3

    def test_incompatible_dims(self):
        comp1 = Component(
            name="A", description="",
            interface_in={}, interface_out={"shape": "[batch, seq_len, d_model]"}
        )
        comp2 = Component(
            name="B", description="",
            interface_in={"shape": "[batch, d_model]"}, interface_out={}
        )
        compatible, score = interfaces_compatible(comp1, comp2)
        assert compatible is False
        assert score == 0.0


class TestGetComponentCategory:
    """Test component category inference."""

    def test_attention_category(self):
        comp = Component(name="MultiHeadAttention", description="", interface_in={}, interface_out={})
        assert get_component_category(comp) == "attention"

    def test_norm_category(self):
        comp = Component(name="LayerNorm", description="", interface_in={}, interface_out={})
        assert get_component_category(comp) == "layer"

    def test_position_category(self):
        # RotaryEmbedding contains "embedding" so gets classified as embedding
        # PositionalEncoding would be position
        comp = Component(name="SinusoidalPosition", description="", interface_in={}, interface_out={})
        assert get_component_category(comp) == "position"

    def test_rope_is_embedding(self):
        # RoPE has "embedding" in name, so classified as embedding (expected behavior)
        comp = Component(name="RotaryEmbedding", description="", interface_in={}, interface_out={})
        assert get_component_category(comp) == "embedding"

    def test_output_category(self):
        comp = Component(name="SoftmaxOutput", description="", interface_in={}, interface_out={})
        assert get_component_category(comp) == "output"

    def test_ssm_category(self):
        comp = Component(name="SelectiveSSM", description="", interface_in={}, interface_out={})
        assert get_component_category(comp) == "attention"

    def test_default_category(self):
        comp = Component(name="UnknownModule", description="", interface_in={}, interface_out={})
        assert get_component_category(comp) == "layer"


class TestGreedyCompose:
    """Test greedy composition strategy."""

    def test_greedy_returns_components(self, db):
        composer = EngineComposer(db)
        components = composer.greedy_compose()
        assert len(components) > 0
        assert len(components) <= 6  # Default max

    def test_greedy_with_start_component(self, db):
        composer = EngineComposer(db)
        components = composer.greedy_compose(start_component="MultiHeadAttention")
        assert len(components) > 0
        # Should include attention-related component
        names = [c.name for c in components]
        assert any("Attention" in n for n in names)

    def test_greedy_sorted_by_architecture_order(self, db):
        composer = EngineComposer(db)
        components = composer.greedy_compose()
        categories = [get_component_category(c) for c in components]
        orders = [CATEGORY_ORDER.get(cat, 4) for cat in categories]
        # Should be sorted (allowing equal values)
        assert orders == sorted(orders)

    def test_greedy_empty_db(self, empty_db):
        composer = EngineComposer(empty_db)
        components = composer.greedy_compose()
        assert components == []


class TestRandomWalkCompose:
    """Test random walk composition strategy."""

    def test_random_returns_components(self, db):
        composer = EngineComposer(db)
        components = composer.random_walk_compose(steps=5)
        assert len(components) > 0
        assert len(components) <= 5

    def test_random_with_temperature(self, db):
        composer = EngineComposer(db)
        # Low temperature should be more deterministic
        components1 = composer.random_walk_compose(steps=3, temperature=0.1)
        assert len(components1) > 0

        # High temperature allows more exploration
        components2 = composer.random_walk_compose(steps=3, temperature=2.0)
        assert len(components2) > 0

    def test_random_empty_db(self, empty_db):
        composer = EngineComposer(empty_db)
        components = composer.random_walk_compose(steps=5)
        assert components == []


class TestCrossover:
    """Test crossover composition strategy."""

    def test_crossover_returns_components(self, db):
        composer = EngineComposer(db)
        components = composer.crossover("Transformer", "Mamba")
        assert len(components) > 0

    def test_crossover_combines_parents(self, db):
        composer = EngineComposer(db)
        components = composer.crossover("Transformer", "LLaMA")
        names = [c.name for c in components]
        # Should have components from both
        assert len(names) >= 2

    def test_crossover_invalid_engine(self, db):
        composer = EngineComposer(db)
        components = composer.crossover("Transformer", "NonexistentEngine")
        assert components == []

    def test_crossover_both_invalid(self, db):
        composer = EngineComposer(db)
        components = composer.crossover("Fake1", "Fake2")
        assert components == []


class TestMutate:
    """Test mutation composition strategy."""

    def test_mutate_returns_components(self, db):
        composer = EngineComposer(db)
        components = composer.mutate("Transformer", mutation_rate=0.5)
        assert len(components) > 0

    def test_mutate_preserves_size(self, db):
        composer = EngineComposer(db)
        transformer = db.get_engine_by_name("Transformer")
        original_size = len(transformer.component_ids)

        components = composer.mutate("Transformer", mutation_rate=0.5)
        # Should have similar number of components
        assert abs(len(components) - original_size) <= 2

    def test_mutate_low_rate(self, db):
        composer = EngineComposer(db)
        # With 0% mutation, should return original components
        components = composer.mutate("Transformer", mutation_rate=0.0)
        assert len(components) > 0

    def test_mutate_invalid_engine(self, db):
        composer = EngineComposer(db)
        components = composer.mutate("NonexistentEngine")
        assert components == []


class TestDream:
    """Test the unified dream interface."""

    def test_dream_greedy(self, db):
        composer = EngineComposer(db)
        components, score = composer.dream("greedy")
        assert len(components) > 0
        assert 0 <= score <= 1.0

    def test_dream_random(self, db):
        composer = EngineComposer(db)
        components, score = composer.dream("random", steps=4)
        assert len(components) > 0
        assert 0 <= score <= 1.0

    def test_dream_crossover(self, db):
        composer = EngineComposer(db)
        components, score = composer.dream("crossover", engine1_name="Transformer", engine2_name="Mamba")
        assert len(components) > 0
        assert 0 <= score <= 1.0

    def test_dream_mutate(self, db):
        composer = EngineComposer(db)
        components, score = composer.dream("mutate", engine_name="Transformer")
        assert len(components) > 0
        assert 0 <= score <= 1.0

    def test_dream_unknown_strategy(self, db):
        composer = EngineComposer(db)
        with pytest.raises(ValueError, match="Unknown strategy"):
            composer.dream("unknown_strategy")

    def test_dream_missing_crossover_args(self, db):
        composer = EngineComposer(db)
        with pytest.raises(ValueError, match="crossover strategy requires"):
            composer.dream("crossover")

    def test_dream_missing_mutate_args(self, db):
        composer = EngineComposer(db)
        with pytest.raises(ValueError, match="mutate strategy requires"):
            composer.dream("mutate")

    def test_dream_failure_returns_negative_score(self, db):
        composer = EngineComposer(db)
        components, score = composer.dream("crossover", engine1_name="Fake1", engine2_name="Fake2")
        assert components == []
        assert score == -1.0


class TestCompatibilityScore:
    """Test compatibility scoring between components."""

    def test_compatibility_uses_interface(self, db):
        composer = EngineComposer(db)
        comp1 = db.find_components("MultiHeadAttention")[0]
        comp2 = db.find_components("FeedForward")[0]
        score = composer.get_compatibility_score(comp1, comp2)
        assert 0 <= score <= 1.0

    def test_compatibility_uses_category_order(self, db):
        composer = EngineComposer(db)
        # Position before attention should score well
        pos = db.find_components("Embedding")[0]
        attn = db.find_components("MultiHeadAttention")[0]
        score_correct = composer.get_compatibility_score(pos, attn)

        # Reverse order should score lower (no category bonus)
        score_reverse = composer.get_compatibility_score(attn, pos)
        assert score_correct >= score_reverse
