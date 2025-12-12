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
from arcfusion.db import Recipe, RecipeAdjustment


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


# Recipe tests
def test_add_recipe(db):
    """Test adding and retrieving a recipe."""
    recipe = Recipe(
        name="TestRecipe",
        component_ids=["comp1", "comp2", "comp3"],
        assembly={
            "connections": [("comp1", "comp2"), ("comp2", "comp3")],
            "residuals": [("comp1", "comp3")],
            "shapes": {"comp1": {"in": "[b,n,d]", "out": "[b,n,d]"}}
        },
        strategy="greedy",
        estimated_score=0.85
    )
    recipe_id = db.add_recipe(recipe)
    assert recipe_id == recipe.recipe_id

    retrieved = db.get_recipe(recipe_id)
    assert retrieved.name == "TestRecipe"
    assert retrieved.component_ids == ["comp1", "comp2", "comp3"]
    assert retrieved.strategy == "greedy"
    assert retrieved.estimated_score == 0.85
    assert len(retrieved.assembly["connections"]) == 2


def test_list_recipes(db):
    """Test listing recipes with filters."""
    # Add multiple recipes
    for i in range(3):
        recipe = Recipe(
            name=f"Recipe{i}",
            component_ids=[f"comp{i}"],
            assembly={"connections": []},
            strategy="greedy" if i % 2 == 0 else "random",
            estimated_score=0.5 + (i * 0.1)
        )
        db.add_recipe(recipe)

    # List all
    all_recipes = db.list_recipes()
    assert len(all_recipes) == 3

    # Filter by strategy
    greedy_recipes = db.list_recipes(strategy="greedy")
    assert len(greedy_recipes) == 2

    # Filter by score
    high_score = db.list_recipes(min_score=0.6)
    assert len(high_score) == 2


def test_delete_recipe(db):
    """Test deleting a recipe and its adjustments."""
    recipe = Recipe(
        name="ToDelete",
        component_ids=["comp1"],
        assembly={},
        strategy="greedy"
    )
    recipe_id = db.add_recipe(recipe)

    # Add an adjustment
    adj = RecipeAdjustment(
        recipe_id=recipe_id,
        adjustment_type="shape_fix",
        original_value="512",
        adjusted_value="128",
        reason="Memory constraint"
    )
    db.add_adjustment(adj)

    # Verify both exist
    assert db.get_recipe(recipe_id) is not None
    assert len(db.get_adjustments(recipe_id)) == 1

    # Delete recipe (should cascade to adjustments)
    result = db.delete_recipe(recipe_id)
    assert result is True
    assert db.get_recipe(recipe_id) is None
    assert len(db.get_adjustments(recipe_id)) == 0


def test_add_adjustment(db):
    """Test adding and retrieving recipe adjustments."""
    recipe = Recipe(
        name="AdjustmentTest",
        component_ids=["comp1"],
        assembly={},
        strategy="greedy"
    )
    recipe_id = db.add_recipe(recipe)

    adj = RecipeAdjustment(
        recipe_id=recipe_id,
        adjustment_type="shape_fix",
        original_value="[batch, seq, 512]",
        adjusted_value="[batch, seq, 128]",
        reason="Reduced d_model to fit memory",
        component_id="comp1"
    )
    adj_id = db.add_adjustment(adj)
    assert adj_id == adj.adjustment_id

    adjustments = db.get_adjustments(recipe_id)
    assert len(adjustments) == 1
    assert adjustments[0].adjustment_type == "shape_fix"
    assert adjustments[0].original_value == "[batch, seq, 512]"
    assert adjustments[0].reason == "Reduced d_model to fit memory"


def test_adjustment_stats(db):
    """Test getting adjustment statistics."""
    recipe = Recipe(
        name="StatsTest",
        component_ids=["comp1"],
        assembly={},
        strategy="greedy"
    )
    recipe_id = db.add_recipe(recipe)

    # Add various adjustment types
    for i, adj_type in enumerate(["shape_fix", "shape_fix", "param_change", "skip_component"]):
        adj = RecipeAdjustment(
            recipe_id=recipe_id,
            adjustment_type=adj_type,
            original_value=f"orig{i}",
            adjusted_value=f"adj{i}",
            reason=f"reason{i}"
        )
        db.add_adjustment(adj)

    stats = db.get_adjustment_stats()
    assert stats["shape_fix"] == 2
    assert stats["param_change"] == 1
    assert stats["skip_component"] == 1


def test_recipe_in_stats(db):
    """Test that recipes appear in DB stats."""
    recipe = Recipe(
        name="StatsRecipe",
        component_ids=["comp1"],
        assembly={},
        strategy="greedy"
    )
    db.add_recipe(recipe)

    adj = RecipeAdjustment(
        recipe_id=recipe.recipe_id,
        adjustment_type="test",
        original_value="a",
        adjusted_value="b",
        reason="test"
    )
    db.add_adjustment(adj)

    stats = db.stats()
    assert stats["recipes"] == 1
    assert stats["recipe_adjustments"] == 1
