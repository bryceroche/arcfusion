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
from arcfusion.db import Recipe, RecipeAdjustment, DreamCandidate


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


# Dream Candidate tests
def test_add_dream_candidate(db):
    """Test adding and retrieving a dream candidate."""
    candidate = DreamCandidate(
        strategy="greedy",
        temperature=0.3,
        components_json='["Attention", "FFN", "LayerNorm"]',
        n_layers=10,
        n_kv_heads=8,
        has_mamba=False,
        has_linear_attn=False,
        is_hybrid=False,
        arch_type="mha",
        predicted_ppl=250.5,
        predicted_time=120.0,
        was_trained=False,
    )
    candidate_id = db.add_dream_candidate(candidate)
    assert candidate_id == candidate.candidate_id

    # Retrieve via list
    candidates = db.list_dream_candidates(limit=10)
    assert len(candidates) == 1
    assert candidates[0].strategy == "greedy"
    assert candidates[0].predicted_ppl == 250.5


def test_list_dream_candidates_filters(db):
    """Test listing dream candidates with various filters."""
    # Add multiple candidates
    for i, arch_type in enumerate(["mha", "gqa", "mamba"]):
        candidate = DreamCandidate(
            strategy="greedy" if i % 2 == 0 else "random",
            temperature=0.1 * i,
            components_json=f'["Component{i}"]',
            n_layers=10,
            n_kv_heads=8 if arch_type != "mamba" else 0,
            has_mamba=arch_type == "mamba",
            has_linear_attn=False,
            is_hybrid=False,
            arch_type=arch_type,
            predicted_ppl=200.0 + i * 10,
            predicted_time=100.0,
            was_trained=i == 1,  # Only second one is trained
        )
        db.add_dream_candidate(candidate)

    # Test untrained_only filter
    untrained = db.list_dream_candidates(untrained_only=True)
    assert len(untrained) == 2
    assert all(not c.was_trained for c in untrained)

    # Test trained_only filter
    trained = db.list_dream_candidates(trained_only=True)
    assert len(trained) == 1
    assert trained[0].was_trained

    # Test arch_type filter
    mamba_candidates = db.list_dream_candidates(arch_type="mamba")
    assert len(mamba_candidates) == 1
    assert mamba_candidates[0].has_mamba


def test_update_dream_candidate_predictions(db):
    """Test updating predictions for a dream candidate."""
    candidate = DreamCandidate(
        strategy="greedy",
        temperature=0.2,
        components_json='["Attention"]',
        n_layers=10,
        n_kv_heads=8,
        has_mamba=False,
        has_linear_attn=False,
        is_hybrid=False,
        arch_type="mha",
        predicted_ppl=300.0,
        predicted_time=150.0,
        was_trained=False,
    )
    candidate_id = db.add_dream_candidate(candidate)

    # Update predictions
    db.update_dream_candidate_predictions(candidate_id, 250.0, 120.0)

    # Verify update
    candidates = db.list_dream_candidates(limit=1)
    assert len(candidates) == 1
    assert candidates[0].predicted_ppl == 250.0
    assert candidates[0].predicted_time == 120.0


def test_update_dream_candidate_training(db):
    """Test updating training results for a dream candidate."""
    candidate = DreamCandidate(
        strategy="greedy",
        temperature=0.2,
        components_json='["Attention"]',
        n_layers=10,
        n_kv_heads=8,
        has_mamba=False,
        has_linear_attn=False,
        is_hybrid=False,
        arch_type="mha",
        predicted_ppl=300.0,
        predicted_time=150.0,
        was_trained=False,
    )
    candidate_id = db.add_dream_candidate(candidate)

    # Update with training results
    db.update_dream_candidate_training(
        candidate_id=candidate_id,
        training_run_id="run-12345",
        actual_ppl=245.0,
        actual_time=135.0
    )

    # Verify update
    candidates = db.list_dream_candidates(trained_only=True)
    assert len(candidates) == 1
    assert candidates[0].was_trained
    assert candidates[0].actual_ppl == 245.0
    assert candidates[0].actual_time == 135.0
    assert candidates[0].training_run_id == "run-12345"
