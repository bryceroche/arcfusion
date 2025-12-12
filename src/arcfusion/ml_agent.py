"""
ML Agent - Execute recipes from Composer with modification tracking.

The ML Agent receives a Recipe from the Composer and:
1. Makes best effort to train the model
2. Stays faithful to the recipe provided
3. Records ANY modifications needed to enable training
4. Enables recreation of training runs

Key principle: ML Agent has leeway to make necessary adjustments,
but ALL adjustments must be recorded in the database.
"""

from dataclasses import dataclass, field
from typing import Optional
import time

from .db import ArcFusionDB, Recipe, RecipeAdjustment, Component
from .codegen import CodeGenerator
from .composer import EngineComposer

# Adjustment type constants
ADJ_SHAPE_FIX = "shape_fix"
ADJ_PARAM_CHANGE = "param_change"
ADJ_COMPONENT_SKIP = "component_skip"
ADJ_COMPONENT_ADD = "component_add"
ADJ_LAYER_REORDER = "layer_reorder"
ADJ_DTYPE_CHANGE = "dtype_change"
ADJ_BUILD_FIX = "build_fix"
ADJ_TRAIN_FIX = "train_fix"

# Try to import validation pipeline (requires torch)
try:
    from .validator import (
        ValidationPipeline,
        ValidationResult,
        ModelConfig,
        TrainingConfig,
    )
    HAS_VALIDATOR = True
except ImportError:
    HAS_VALIDATOR = False
    ValidationPipeline = None
    ValidationResult = None
    ModelConfig = None
    TrainingConfig = None


@dataclass
class ExecutionResult:
    """Result of ML Agent executing a recipe."""
    recipe_id: str
    success: bool
    # Model info
    model_name: str = ""
    num_parameters: int = 0
    # Training results
    final_loss: float = float('inf')
    perplexity: float = float('inf')
    training_steps: int = 0
    training_time_seconds: float = 0.0
    # Adjustments made
    adjustments: list[RecipeAdjustment] = field(default_factory=list)
    num_adjustments: int = 0
    # Errors (if any)
    build_error: Optional[str] = None
    train_error: Optional[str] = None
    # Generated code
    generated_code: Optional[str] = None
    # Benchmarks
    benchmarks: dict = field(default_factory=dict)

    @property
    def is_faithful(self) -> bool:
        """True if no adjustments were needed."""
        return self.num_adjustments == 0

    @property
    def adjustment_summary(self) -> str:
        """Human-readable summary of adjustments."""
        if not self.adjustments:
            return "No adjustments needed"
        types = {}
        for adj in self.adjustments:
            types[adj.adjustment_type] = types.get(adj.adjustment_type, 0) + 1
        parts = [f"{count} {atype}" for atype, count in types.items()]
        return ", ".join(parts)


class MLAgent:
    """
    ML Agent that executes recipes from Composer.

    Responsibilities:
    - Execute recipes faithfully
    - Track any necessary modifications
    - Store results for reproducibility
    - Provide feedback for Composer improvement
    """

    def __init__(
        self,
        db: ArcFusionDB,
        model_config: Optional['ModelConfig'] = None,
        training_config: Optional['TrainingConfig'] = None,
    ):
        if not HAS_VALIDATOR:
            raise ImportError("MLAgent requires PyTorch. Install with: pip install torch")

        self.db = db
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        self.composer = EngineComposer(db)
        self.codegen = CodeGenerator(db)
        self.pipeline = ValidationPipeline(
            db=db,
            model_config=self.model_config,
            training_config=self.training_config,
        )

    def execute_recipe(
        self,
        recipe: Recipe,
        verbose: bool = True,
        store_results: bool = True,
    ) -> ExecutionResult:
        """
        Execute a recipe: build, train, and benchmark the model.

        Args:
            recipe: Recipe from Composer
            verbose: Print progress
            store_results: Store adjustments in database

        Returns:
            ExecutionResult with training metrics and adjustment history
        """
        result = ExecutionResult(
            recipe_id=recipe.recipe_id,
            success=False,
            model_name=recipe.name,
        )
        adjustments = []

        if verbose:
            print(f"ML Agent executing recipe: {recipe.name}")
            print(f"  Strategy: {recipe.strategy}")
            print(f"  Components: {len(recipe.component_ids)}")
            print(f"  Estimated score: {recipe.estimated_score:.2f}")
            print()

        # Step 1: Get components from recipe
        components = self.composer.recipe_to_components(recipe)

        if len(components) != len(recipe.component_ids):
            # Some components missing - record adjustment
            missing = set(recipe.component_ids) - {c.component_id for c in components}
            for cid in missing:
                adj = RecipeAdjustment(
                    recipe_id=recipe.recipe_id,
                    adjustment_type=ADJ_COMPONENT_SKIP,
                    original_value=cid,
                    adjusted_value="(skipped)",
                    reason=f"Component {cid} not found in database",
                )
                adjustments.append(adj)
                if verbose:
                    print(f"  [ADJ] Skipping missing component: {cid}")

        if not components:
            result.build_error = "No valid components found"
            result.adjustments = adjustments
            result.num_adjustments = len(adjustments)
            # Store adjustments even on early failure
            if store_results:
                for adj in adjustments:
                    self.db.add_adjustment(adj)
            return result

        # Step 2: Generate code
        if verbose:
            print("Generating code...")

        generated = self.codegen.generate(
            components=components,
            name=recipe.name,
            description=f"ML Agent execution of recipe {recipe.recipe_id}"
        )

        # Validate syntax
        valid, error = generated.validate_syntax()
        if not valid:
            # Try to fix common issues
            adj = RecipeAdjustment(
                recipe_id=recipe.recipe_id,
                adjustment_type=ADJ_BUILD_FIX,
                original_value="generated code",
                adjusted_value="syntax error",
                reason=f"Code generation produced invalid syntax: {error}",
            )
            adjustments.append(adj)
            result.build_error = f"Syntax error: {error}"
            result.adjustments = adjustments
            result.num_adjustments = len(adjustments)
            result.generated_code = generated.code
            if verbose:
                print(f"  [ADJ] Build failed: {error}")
            # Store adjustments even on early failure
            if store_results:
                for adj in adjustments:
                    self.db.add_adjustment(adj)
            return result

        result.generated_code = generated.code

        # Step 3: Run validation pipeline
        if verbose:
            print("Running validation pipeline...")

        start_time = time.time()
        validation = self.pipeline.validate(generated, verbose=verbose)
        total_time = time.time() - start_time

        # Check for build errors
        if validation.build_error:
            adj = RecipeAdjustment(
                recipe_id=recipe.recipe_id,
                adjustment_type=ADJ_BUILD_FIX,
                original_value="model build",
                adjusted_value="failed",
                reason=validation.build_error,
            )
            adjustments.append(adj)
            result.build_error = validation.build_error
            if verbose:
                print(f"  [ADJ] Build error: {validation.build_error}")

        # Check for training errors
        if validation.train_error:
            adj = RecipeAdjustment(
                recipe_id=recipe.recipe_id,
                adjustment_type=ADJ_TRAIN_FIX,
                original_value="training",
                adjusted_value="failed",
                reason=validation.train_error,
            )
            adjustments.append(adj)
            result.train_error = validation.train_error
            if verbose:
                print(f"  [ADJ] Train error: {validation.train_error}")

        # Record parameter adjustments if d_model or vocab_size changed
        if hasattr(self.pipeline, 'model_config'):
            cfg = self.pipeline.model_config
            # Check if we used different params than assembly suggested
            assembly_shapes = recipe.assembly.get('shapes', {})
            for cid, shapes in assembly_shapes.items():
                if 'd_model' in str(shapes):
                    # Assembly specified a d_model - check if we changed it
                    # This is a simplified check; real impl would parse the shape
                    pass

        # Populate result
        result.success = validation.success
        result.num_parameters = validation.num_parameters
        result.final_loss = validation.final_loss
        result.perplexity = validation.perplexity
        result.training_steps = validation.training_steps
        result.training_time_seconds = total_time
        result.benchmarks = validation.benchmarks
        result.adjustments = adjustments
        result.num_adjustments = len(adjustments)

        # Step 4: Store adjustments in database
        if store_results:
            for adj in adjustments:
                self.db.add_adjustment(adj)
            if verbose and adjustments:
                print(f"\nStored {len(adjustments)} adjustments in database")

        # Summary
        if verbose:
            print(f"\n{'='*60}")
            print("Execution Summary")
            print(f"{'='*60}")
            print(f"  Success: {'Yes' if result.success else 'No'}")
            print(f"  Faithful: {'Yes' if result.is_faithful else 'No'}")
            print(f"  Adjustments: {result.adjustment_summary}")
            if result.success:
                print(f"  Parameters: {result.num_parameters:,}")
                print(f"  Final Loss: {result.final_loss:.4f}")
                print(f"  Perplexity: {result.perplexity:.2f}")
                print(f"  Time: {result.training_time_seconds:.1f}s")

        return result

    def execute_recipe_by_id(
        self,
        recipe_id: str,
        verbose: bool = True,
        store_results: bool = True,
    ) -> ExecutionResult:
        """Execute a recipe by its ID."""
        recipe = self.db.get_recipe(recipe_id)
        if not recipe:
            return ExecutionResult(
                recipe_id=recipe_id,
                success=False,
                build_error=f"Recipe {recipe_id} not found",
            )
        return self.execute_recipe(recipe, verbose=verbose, store_results=store_results)

    def dream_and_execute(
        self,
        name: str,
        strategy: str = "greedy",
        verbose: bool = True,
        store_results: bool = True,
        **kwargs
    ) -> tuple[Recipe, ExecutionResult]:
        """
        Dream up a recipe and immediately execute it.

        Convenience method that combines Composer.create_recipe() with execute_recipe().

        Args:
            name: Name for the recipe/model
            strategy: Dream strategy (greedy, random, crossover, mutate)
            verbose: Print progress
            store_results: Store recipe and adjustments
            **kwargs: Additional args for dream strategy

        Returns:
            (Recipe, ExecutionResult) tuple
        """
        if verbose:
            print(f"Dreaming recipe with {strategy} strategy...")

        recipe = self.composer.create_recipe(
            name=name,
            strategy=strategy,
            save_to_db=store_results,
            **kwargs
        )

        if verbose:
            print(f"Created recipe: {recipe.recipe_id}")
            print()

        result = self.execute_recipe(
            recipe,
            verbose=verbose,
            store_results=store_results,
        )

        return recipe, result

    def get_adjustment_feedback(self, limit: int = 100) -> dict:
        """
        Get feedback from adjustments for Composer improvement.

        Analyzes stored adjustments to identify common issues that
        the Composer should learn from.

        Returns:
            Dict with adjustment statistics and recommendations
        """
        stats = self.db.get_adjustment_stats()

        # Get recent adjustments for each type
        feedback = {
            'total_adjustments': sum(stats.values()),
            'by_type': stats,
            'recommendations': [],
        }

        # Generate recommendations based on patterns
        if stats.get(ADJ_COMPONENT_SKIP, 0) > 5:
            feedback['recommendations'].append(
                "Many components being skipped - check DB for missing components"
            )

        if stats.get(ADJ_SHAPE_FIX, 0) > 5:
            feedback['recommendations'].append(
                "Frequent shape fixes - Composer may need better interface matching"
            )

        if stats.get(ADJ_BUILD_FIX, 0) > 3:
            feedback['recommendations'].append(
                "Build failures occurring - review code generation patterns"
            )

        if stats.get(ADJ_TRAIN_FIX, 0) > 3:
            feedback['recommendations'].append(
                "Training failures occurring - check model compatibility"
            )

        return feedback
