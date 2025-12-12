"""Tests for the ML Agent."""

import pytest
from arcfusion import (
    ArcFusionDB,
    Recipe,
    RecipeAdjustment,
    Component,
)
from arcfusion.seeds import seed_transformers

# Skip all tests if torch not available
pytest.importorskip("torch")

from arcfusion import MLAgent, ExecutionResult, ModelConfig, TrainingConfig


@pytest.fixture
def db():
    """Create a test database with transformer components."""
    db = ArcFusionDB(":memory:")
    seed_transformers(db)
    return db


@pytest.fixture
def agent(db):
    """Create an ML Agent with small config for fast tests."""
    model_cfg = ModelConfig(d_model=64, vocab_size=100, max_seq_len=16)
    train_cfg = TrainingConfig(batch_size=2, max_steps=5, device='cpu')
    return MLAgent(db, model_config=model_cfg, training_config=train_cfg)


@pytest.fixture
def simple_recipe(db):
    """Create a simple valid recipe."""
    components = db.find_components('')
    # Get Embedding, Attention, and Output components
    selected = []
    for name in ['Embedding', 'MultiHeadAttention', 'SoftmaxOutput']:
        matches = [c for c in components if c.name == name]
        if matches:
            selected.append(matches[0])

    recipe = Recipe(
        name='TestRecipe',
        component_ids=[c.component_id for c in selected],
        assembly={'connections': [], 'residuals': [], 'shapes': {}, 'categories': {}, 'notes': []},
        strategy='test',
        estimated_score=0.8
    )
    db.add_recipe(recipe)
    return recipe


class TestExecutionResult:
    """Test ExecutionResult dataclass."""

    def test_is_faithful_when_no_adjustments(self):
        result = ExecutionResult(
            recipe_id="test",
            success=True,
            adjustments=[],
            num_adjustments=0
        )
        assert result.is_faithful is True

    def test_not_faithful_when_adjustments(self):
        adj = RecipeAdjustment(
            recipe_id="test",
            adjustment_type="test",
            original_value="a",
            adjusted_value="b",
            reason="test"
        )
        result = ExecutionResult(
            recipe_id="test",
            success=True,
            adjustments=[adj],
            num_adjustments=1
        )
        assert result.is_faithful is False

    def test_adjustment_summary_no_adjustments(self):
        result = ExecutionResult(
            recipe_id="test",
            success=True,
            adjustments=[],
            num_adjustments=0
        )
        assert result.adjustment_summary == "No adjustments needed"

    def test_adjustment_summary_with_adjustments(self):
        adjs = [
            RecipeAdjustment(
                recipe_id="test",
                adjustment_type="shape_fix",
                original_value="a",
                adjusted_value="b",
                reason="test"
            ),
            RecipeAdjustment(
                recipe_id="test",
                adjustment_type="shape_fix",
                original_value="c",
                adjusted_value="d",
                reason="test"
            ),
            RecipeAdjustment(
                recipe_id="test",
                adjustment_type="param_change",
                original_value="e",
                adjusted_value="f",
                reason="test"
            ),
        ]
        result = ExecutionResult(
            recipe_id="test",
            success=True,
            adjustments=adjs,
            num_adjustments=3
        )
        summary = result.adjustment_summary
        assert "2 shape_fix" in summary
        assert "1 param_change" in summary


class TestMLAgent:
    """Test MLAgent class."""

    def test_init(self, db):
        agent = MLAgent(db)
        assert agent.db == db
        assert agent.model_config is not None
        assert agent.training_config is not None

    def test_init_with_custom_config(self, db):
        model_cfg = ModelConfig(d_model=256)
        train_cfg = TrainingConfig(max_steps=50)
        agent = MLAgent(db, model_config=model_cfg, training_config=train_cfg)
        assert agent.model_config.d_model == 256
        assert agent.training_config.max_steps == 50

    def test_execute_recipe_returns_result(self, agent, simple_recipe):
        result = agent.execute_recipe(simple_recipe, verbose=False, store_results=False)
        assert isinstance(result, ExecutionResult)
        assert result.recipe_id == simple_recipe.recipe_id

    def test_execute_recipe_generates_code(self, agent, simple_recipe):
        result = agent.execute_recipe(simple_recipe, verbose=False, store_results=False)
        assert result.generated_code is not None
        assert len(result.generated_code) > 0
        assert "class TestRecipe" in result.generated_code

    def test_execute_recipe_by_id(self, agent, simple_recipe, db):
        result = agent.execute_recipe_by_id(simple_recipe.recipe_id, verbose=False, store_results=False)
        assert result.recipe_id == simple_recipe.recipe_id

    def test_execute_recipe_by_id_not_found(self, agent):
        result = agent.execute_recipe_by_id("nonexistent", verbose=False, store_results=False)
        assert result.success is False
        assert "not found" in result.build_error

    def test_execute_recipe_stores_adjustments(self, agent, db):
        # Create a recipe with missing component to trigger adjustment
        recipe = Recipe(
            name='MissingComponentRecipe',
            component_ids=['nonexistent123'],
            assembly={},
            strategy='test'
        )
        db.add_recipe(recipe)

        result = agent.execute_recipe(recipe, verbose=False, store_results=True)

        # Check adjustment was stored
        adjustments = db.get_adjustments(recipe.recipe_id)
        assert len(adjustments) >= 1
        assert any(adj.adjustment_type == "component_skip" for adj in adjustments)

    def test_execute_recipe_no_store(self, agent, db):
        # Create a recipe with missing component
        recipe = Recipe(
            name='NoStoreRecipe',
            component_ids=['nonexistent456'],
            assembly={},
            strategy='test'
        )
        db.add_recipe(recipe)

        result = agent.execute_recipe(recipe, verbose=False, store_results=False)

        # Adjustments should NOT be stored
        adjustments = db.get_adjustments(recipe.recipe_id)
        assert len(adjustments) == 0

    def test_dream_and_execute(self, db):
        # Use larger d_model to match generated code defaults
        model_cfg = ModelConfig(d_model=512, vocab_size=100, max_seq_len=16)
        train_cfg = TrainingConfig(batch_size=2, max_steps=3, device='cpu')
        agent = MLAgent(db, model_config=model_cfg, training_config=train_cfg)

        recipe, result = agent.dream_and_execute(
            name='DreamTest',
            strategy='random',
            steps=3,
            verbose=False,
            store_results=False
        )

        assert isinstance(recipe, Recipe)
        assert isinstance(result, ExecutionResult)
        assert recipe.name == 'DreamTest'
        assert recipe.strategy == 'random'

    def test_get_adjustment_feedback(self, agent, db):
        # Add some adjustments
        for i in range(3):
            adj = RecipeAdjustment(
                recipe_id=f"test{i}",
                adjustment_type="shape_fix",
                original_value="a",
                adjusted_value="b",
                reason="test"
            )
            db.add_adjustment(adj)

        feedback = agent.get_adjustment_feedback()

        assert 'total_adjustments' in feedback
        assert 'by_type' in feedback
        assert 'recommendations' in feedback
        assert feedback['total_adjustments'] == 3
        assert feedback['by_type'].get('shape_fix') == 3


class TestMLAgentMissingComponents:
    """Test ML Agent handling of missing/invalid components."""

    def test_missing_component_recorded(self, agent, db):
        recipe = Recipe(
            name='MissingTest',
            component_ids=['missing1', 'missing2'],
            assembly={},
            strategy='test'
        )
        db.add_recipe(recipe)

        result = agent.execute_recipe(recipe, verbose=False, store_results=False)

        # Should record skip adjustments
        assert result.num_adjustments >= 2
        assert any(adj.adjustment_type == "component_skip" for adj in result.adjustments)

    def test_empty_components_fails(self, agent, db):
        recipe = Recipe(
            name='EmptyTest',
            component_ids=[],
            assembly={},
            strategy='test'
        )
        db.add_recipe(recipe)

        result = agent.execute_recipe(recipe, verbose=False, store_results=False)

        assert result.success is False
        assert result.build_error is not None
