"""Tests for ArcFusionDB."""

import os
import tempfile
import pytest
from arcfusion import (
    ArcFusionDB,
    Component,
    Engine,
    BenchmarkResult,
    ProcessedPaper,
    seed_transformers,
    seed_modern_architectures,
)


@pytest.fixture
def db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = ArcFusionDB(path)
    yield database
    database.close()
    os.unlink(path)


@pytest.fixture
def seeded_db(db):
    """Database with seed data."""
    seed_transformers(db, verbose=False)
    seed_modern_architectures(db, verbose=False)
    return db


def test_add_component(db):
    comp = Component(
        name="TestComponent",
        description="A test component",
        interface_in={"shape": "[batch, seq]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq]", "dtype": "float32"},
        usefulness_score=0.75
    )
    comp_id = db.add_component(comp)
    assert comp_id == comp.component_id

    retrieved = db.get_component(comp_id)
    assert retrieved.name == "TestComponent"
    assert retrieved.usefulness_score == 0.75


def test_find_components(seeded_db):
    # Find by name
    attention = seeded_db.find_components("Attention")
    assert len(attention) > 0
    assert "Attention" in attention[0].name

    # Find by score
    high_score = seeded_db.find_components(min_score=0.9)
    assert all(c.usefulness_score >= 0.9 for c in high_score)


def test_add_engine(db):
    # Add components first
    comp1 = Component(name="A", description="", interface_in={}, interface_out={})
    comp2 = Component(name="B", description="", interface_in={}, interface_out={})
    db.add_component(comp1)
    db.add_component(comp2)

    engine = Engine(
        name="TestEngine",
        description="A test engine",
        component_ids=[comp1.component_id, comp2.component_id],
        engine_score=0.8
    )
    db.add_engine(engine)

    retrieved = db.get_engine_by_name("TestEngine")
    assert retrieved.name == "TestEngine"
    assert len(retrieved.component_ids) == 2


def test_seeded_engines(seeded_db):
    transformer = seeded_db.get_engine_by_name("Transformer")
    assert transformer is not None
    assert len(transformer.component_ids) >= 5

    mamba = seeded_db.get_engine_by_name("Mamba")
    assert mamba is not None


def test_compatible_components(seeded_db):
    attention = seeded_db.find_components("MultiHeadAttention")[0]
    compatible = seeded_db.get_compatible_components(attention.component_id, min_score=0.8)
    assert len(compatible) > 0


def test_processed_papers(db):
    paper = ProcessedPaper(
        arxiv_id="2312.00752",
        title="Mamba Paper",
        status="processed"
    )
    db.add_processed_paper(paper)

    # Check deduplication
    assert db.is_paper_processed("2312.00752")
    assert db.is_paper_processed("https://arxiv.org/abs/2312.00752")
    assert db.is_paper_processed("2312.00752v2")
    assert not db.is_paper_processed("9999.99999")


def test_benchmarks(seeded_db):
    transformer = seeded_db.get_engine_by_name("Transformer")

    result = BenchmarkResult(
        engine_id=transformer.engine_id,
        benchmark_name="perplexity",
        score=18.5,
        parameters={"size": "125M"}
    )
    seeded_db.add_benchmark(result)

    benchmarks = seeded_db.get_engine_benchmarks(transformer.engine_id)
    assert len(benchmarks) == 1
    assert benchmarks[0].score == 18.5


def test_stats(seeded_db):
    stats = seeded_db.stats()
    assert stats['components'] > 0
    assert stats['engines'] > 0
