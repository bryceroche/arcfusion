"""
Tests for the decomposer module - paper analysis and component extraction.
"""

import pytest
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from arcfusion.db import ArcFusionDB, Component
from arcfusion.decomposer import PaperDecomposer
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


@pytest.fixture
def seeded_db():
    """Create a test database with seed data."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    db = ArcFusionDB(db_path)
    seed_transformers(db, verbose=False)
    yield db
    db.close()
    os.unlink(db_path)


class TestComponentPatterns:
    """Test component pattern definitions."""

    def test_patterns_defined(self):
        patterns = PaperDecomposer.COMPONENT_PATTERNS
        assert "attention" in patterns
        assert "normalization" in patterns
        assert "feedforward" in patterns
        assert "ssm" in patterns

    def test_patterns_have_variants(self):
        patterns = PaperDecomposer.COMPONENT_PATTERNS
        assert len(patterns["attention"]) >= 2
        assert "self-attention" in patterns["attention"]

    def test_category_search_patterns_defined(self):
        search = PaperDecomposer.CATEGORY_SEARCH_PATTERNS
        assert "attention" in search
        assert "normalization" in search
        # Should map to actual component name patterns
        assert "Norm" in search["normalization"]


class TestExtractComponents:
    """Test component extraction from text."""

    def test_extract_attention(self, db):
        decomposer = PaperDecomposer(db)
        text = "We use multi-head attention with 8 heads."
        extracted = decomposer.extract_components_from_text(text)

        categories = [e["category"] for e in extracted]
        assert "attention" in categories

    def test_extract_normalization(self, db):
        decomposer = PaperDecomposer(db)
        text = "Layer normalization is applied after each block."
        extracted = decomposer.extract_components_from_text(text)

        categories = [e["category"] for e in extracted]
        assert "normalization" in categories

    def test_extract_multiple(self, db):
        decomposer = PaperDecomposer(db)
        text = """
        We introduce a transformer with multi-head attention,
        layer normalization, and feed-forward networks.
        """
        extracted = decomposer.extract_components_from_text(text)

        categories = [e["category"] for e in extracted]
        assert len(categories) >= 3
        assert "attention" in categories
        assert "normalization" in categories
        assert "feedforward" in categories

    def test_extract_ssm(self, db):
        decomposer = PaperDecomposer(db)
        text = "Mamba uses selective state space models instead of attention."
        extracted = decomposer.extract_components_from_text(text)

        categories = [e["category"] for e in extracted]
        assert "ssm" in categories

    def test_confidence_increases_with_mentions(self, db):
        decomposer = PaperDecomposer(db)

        text_few = "The model uses attention."
        text_many = "attention attention attention attention attention attention attention attention attention attention"

        ext_few = decomposer.extract_components_from_text(text_few)
        ext_many = decomposer.extract_components_from_text(text_many)

        conf_few = next(e["confidence"] for e in ext_few if e["category"] == "attention")
        conf_many = next(e["confidence"] for e in ext_many if e["category"] == "attention")

        assert conf_many > conf_few

    def test_extract_empty_text(self, db):
        decomposer = PaperDecomposer(db)
        extracted = decomposer.extract_components_from_text("")
        assert extracted == []

    def test_extract_no_matches(self, db):
        decomposer = PaperDecomposer(db)
        text = "This paper discusses bananas and oranges."
        extracted = decomposer.extract_components_from_text(text)
        assert len(extracted) == 0


class TestCreateEngineFromPaper:
    """Test engine creation from papers."""

    def test_creates_engine(self, db):
        decomposer = PaperDecomposer(db)
        engine, components = decomposer.create_engine_from_paper(
            title="Test Transformer",
            abstract="A model with multi-head attention and layer normalization.",
            paper_url="https://arxiv.org/abs/1234.5678"
        )

        assert engine is not None
        assert engine.name == "Test Transformer"
        assert "https://arxiv.org/abs/1234.5678" == engine.paper_url

    def test_creates_components(self, db):
        decomposer = PaperDecomposer(db)
        engine, components = decomposer.create_engine_from_paper(
            title="New Architecture",
            abstract="Uses attention and feedforward layers."
        )

        # Should have created some components
        all_components = db.find_components()
        assert len(all_components) >= 2

    def test_reuses_existing_components(self, seeded_db):
        decomposer = PaperDecomposer(seeded_db)
        initial_count = len(seeded_db.find_components())

        engine, new_components = decomposer.create_engine_from_paper(
            title="Another Transformer",
            abstract="Uses attention mechanism."
        )

        # Should reuse existing Attention component, not create new
        final_count = len(seeded_db.find_components())
        # Should have created fewer components than extracted
        assert final_count <= initial_count + 3

    def test_creates_relationships(self, db):
        decomposer = PaperDecomposer(db)
        engine, _ = decomposer.create_engine_from_paper(
            title="Test",
            abstract="Uses attention, normalization, and feedforward."
        )

        stats = db.stats()
        assert stats["relationships"] > 0

    def test_engine_links_to_components(self, db):
        decomposer = PaperDecomposer(db)
        # Use text with multiple clear mentions
        engine, _ = decomposer.create_engine_from_paper(
            title="Test",
            abstract="The model uses multi-head attention for sequence processing. Layer normalization is applied after each sublayer."
        )

        assert len(engine.component_ids) >= 2


class TestCategorySearchPatterns:
    """Test that category search patterns work correctly."""

    def test_attention_pattern_finds_multihead(self, seeded_db):
        """Searching with 'Attention' should find MultiHeadAttention."""
        matches = seeded_db.find_components("Attention")
        names = [m.name for m in matches]
        assert "MultiHeadAttention" in names

    def test_norm_pattern_finds_layernorm(self, seeded_db):
        """Searching with 'Norm' should find LayerNorm."""
        matches = seeded_db.find_components("Norm")
        names = [m.name for m in matches]
        assert "LayerNorm" in names

    def test_feed_pattern_finds_feedforward(self, seeded_db):
        """Searching with 'Feed' should find FeedForward."""
        matches = seeded_db.find_components("Feed")
        names = [m.name for m in matches]
        assert "FeedForward" in names
