"""Tests for ArxivFetcher and PaperDecomposer."""

import os
import tempfile
import pytest
from arcfusion import (
    ArcFusionDB,
    ArxivFetcher,
    ArxivPaper,
    PaperDecomposer,
    seed_transformers,
)


@pytest.fixture
def db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = ArcFusionDB(path)
    seed_transformers(database, verbose=False)
    yield database
    database.close()
    os.unlink(path)


@pytest.fixture
def fetcher(db):
    """Create an ArxivFetcher instance."""
    return ArxivFetcher(db)


@pytest.fixture
def decomposer(db):
    """Create a PaperDecomposer instance."""
    return PaperDecomposer(db)


class TestPaperDecomposer:
    """Tests for the PaperDecomposer class."""

    def test_extract_attention(self, decomposer):
        """Test extraction of attention-related components."""
        text = """
        We propose a novel multi-head attention mechanism that combines
        self-attention with cross-attention for better sequence modeling.
        """
        components = decomposer.extract_components_from_text(text)
        categories = [c["category"] for c in components]
        assert "attention" in categories

    def test_extract_normalization(self, decomposer):
        """Test extraction of normalization components."""
        text = """
        Our model uses RMSNorm instead of LayerNorm for better training stability.
        We also experimented with batch normalization but found layer norm superior.
        """
        components = decomposer.extract_components_from_text(text)
        categories = [c["category"] for c in components]
        assert "normalization" in categories

    def test_extract_ssm(self, decomposer):
        """Test extraction of state space model components."""
        text = """
        Mamba introduces a selective state space model (SSM) that achieves
        linear time complexity while maintaining strong modeling capabilities.
        """
        components = decomposer.extract_components_from_text(text)
        categories = [c["category"] for c in components]
        assert "ssm" in categories

    def test_extract_multiple(self, decomposer):
        """Test extraction of multiple component types."""
        text = """
        The transformer architecture consists of multi-head attention layers,
        feed-forward networks, layer normalization, and residual connections.
        We use GELU activation and dropout for regularization.
        """
        components = decomposer.extract_components_from_text(text)
        categories = [c["category"] for c in components]

        assert "attention" in categories
        assert "feedforward" in categories
        assert "normalization" in categories
        assert "residual" in categories
        assert "activation" in categories
        assert "dropout" in categories

    def test_confidence_scoring(self, decomposer):
        """Test that repeated mentions increase confidence."""
        # Few mentions
        text_low = "attention mechanism"
        # Many mentions
        text_high = " ".join(["attention"] * 15)

        low = decomposer.extract_components_from_text(text_low)
        high = decomposer.extract_components_from_text(text_high)

        low_conf = next(c["confidence"] for c in low if c["category"] == "attention")
        high_conf = next(c["confidence"] for c in high if c["category"] == "attention")

        assert high_conf > low_conf

    def test_create_engine_from_paper(self, decomposer, db):
        """Test creating an engine from paper text."""
        engine, new_components = decomposer.create_engine_from_paper(
            title="Test Transformer Model",
            abstract="A novel transformer with multi-head attention and feed-forward layers.",
            paper_url="https://arxiv.org/abs/0000.00000"
        )

        assert engine is not None
        assert engine.name == "Test Transformer Model"
        assert len(engine.component_ids) > 0

        # Verify engine was added to DB
        retrieved = db.get_engine_by_name("Test Transformer Model")
        assert retrieved is not None


class TestArxivFetcher:
    """Tests for the ArxivFetcher class."""

    def test_normalize_arxiv_id(self, fetcher):
        """Test arXiv ID normalization."""
        # Basic ID
        assert fetcher._normalize_arxiv_id("2312.00752") == "2312.00752"
        # With version
        assert fetcher._normalize_arxiv_id("2312.00752v2") == "2312.00752"
        # URL format
        assert fetcher._normalize_arxiv_id("http://arxiv.org/abs/2312.00752v1") == "2312.00752"

    def test_arxiv_paper_dataclass(self):
        """Test ArxivPaper dataclass."""
        paper = ArxivPaper(
            arxiv_id="2312.00752",
            title="Test Paper",
            abstract="Test abstract",
            authors=["Author One", "Author Two"],
            pdf_url="https://arxiv.org/pdf/2312.00752.pdf",
            published="2023-12-01T00:00:00",
            categories=["cs.LG", "cs.CL"]
        )

        assert paper.arxiv_id == "2312.00752"
        assert len(paper.authors) == 2
        assert len(paper.categories) == 2

    def test_ingest_paper_deduplication(self, fetcher, db):
        """Test that papers are not processed twice."""
        paper = ArxivPaper(
            arxiv_id="test.12345",
            title="Test Paper for Dedup",
            abstract="Testing attention mechanism",
            authors=["Test Author"],
            pdf_url="https://example.com/test.pdf",
            published="2024-01-01T00:00:00",
            categories=["cs.LG"]
        )

        # First ingestion should succeed
        result1 = fetcher.ingest_paper(paper, verbose=False)
        assert result1["status"] == "processed"

        # Second ingestion should be skipped
        result2 = fetcher.ingest_paper(paper, verbose=False)
        assert result2["status"] == "skipped"
        assert result2["reason"] == "already_processed"

    def test_ml_categories_defined(self, fetcher):
        """Test that ML categories are properly defined."""
        assert "cs.LG" in fetcher.ML_CATEGORIES
        assert "cs.CL" in fetcher.ML_CATEGORIES
        assert "cs.CV" in fetcher.ML_CATEGORIES

    def test_architecture_terms_defined(self, fetcher):
        """Test that architecture search terms are defined."""
        assert "transformer" in fetcher.ARCHITECTURE_TERMS
        assert "attention mechanism" in fetcher.ARCHITECTURE_TERMS
        assert "mamba" in fetcher.ARCHITECTURE_TERMS


class TestEngineComposer:
    """Tests for the EngineComposer class."""

    def test_greedy_compose(self, db):
        """Test greedy composition strategy."""
        from arcfusion import EngineComposer, seed_modern_architectures
        seed_modern_architectures(db, verbose=False)

        composer = EngineComposer(db)
        components, score = composer.dream("greedy")

        assert len(components) > 0
        assert 0 <= score <= 1

    def test_random_walk(self, db):
        """Test random walk composition."""
        from arcfusion import EngineComposer, seed_modern_architectures
        seed_modern_architectures(db, verbose=False)

        composer = EngineComposer(db)
        components, score = composer.dream("random", steps=3, temperature=1.0)

        assert len(components) > 0

    def test_crossover(self, db):
        """Test crossover composition."""
        from arcfusion import EngineComposer, seed_modern_architectures
        seed_modern_architectures(db, verbose=False)

        composer = EngineComposer(db)
        components, score = composer.dream(
            "crossover",
            engine1_name="Transformer",
            engine2_name="Mamba"
        )

        assert len(components) > 0

    def test_mutate(self, db):
        """Test mutation composition."""
        from arcfusion import EngineComposer, seed_modern_architectures
        seed_modern_architectures(db, verbose=False)

        composer = EngineComposer(db)
        components, score = composer.dream(
            "mutate",
            engine_name="Transformer",
            mutation_rate=0.5
        )

        assert len(components) > 0
