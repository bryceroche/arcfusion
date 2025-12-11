"""Tests for the validation pipeline module."""

import pytest
from arcfusion import HAS_VALIDATOR
from arcfusion.codegen import GeneratedCode

# Skip all tests if torch is not available
pytestmark = pytest.mark.skipif(not HAS_VALIDATOR, reason="PyTorch not installed")


class TestModelConfig:
    """Test ModelConfig dataclass."""

    @pytest.fixture
    def model_config(self):
        from arcfusion.validator import ModelConfig
        return ModelConfig()

    def test_default_values(self, model_config):
        """Test default configuration values."""
        assert model_config.d_model == 128
        assert model_config.vocab_size == 1000
        assert model_config.max_seq_len == 64
        assert model_config.n_heads == 4
        assert model_config.n_layers == 2
        assert model_config.dropout == 0.1

    def test_custom_values(self):
        """Test custom configuration values."""
        from arcfusion.validator import ModelConfig
        config = ModelConfig(d_model=256, vocab_size=5000, n_heads=8)
        assert config.d_model == 256
        assert config.vocab_size == 5000
        assert config.n_heads == 8


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    @pytest.fixture
    def training_config(self):
        from arcfusion.validator import TrainingConfig
        return TrainingConfig()

    def test_default_values(self, training_config):
        """Test default training configuration values."""
        assert training_config.batch_size == 8
        assert training_config.max_steps == 100
        assert training_config.learning_rate == 1e-4
        assert training_config.max_time_seconds == 300.0
        assert training_config.eval_interval == 50
        assert training_config.device == "cpu"


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_failed_result(self):
        """Test a failed validation result."""
        from arcfusion.validator import ValidationResult
        result = ValidationResult(
            success=False,
            model_name="TestModel",
            build_error="Syntax error"
        )
        assert not result.success
        assert not result.is_trainable
        assert result.build_error == "Syntax error"

    def test_successful_result(self):
        """Test a successful validation result."""
        from arcfusion.validator import ValidationResult
        result = ValidationResult(
            success=True,
            model_name="TestModel",
            num_parameters=1000,
            final_loss=2.5,
            perplexity=12.18,
            training_steps=100,
            training_time_seconds=5.0,
        )
        assert result.success
        assert result.is_trainable
        assert result.num_parameters == 1000
        assert result.final_loss == 2.5


class TestModelBuilder:
    """Test ModelBuilder class."""

    @pytest.fixture
    def builder(self):
        from arcfusion.validator import ModelBuilder
        return ModelBuilder()

    def test_build_invalid_syntax(self, builder):
        """Test building code with syntax errors."""
        generated = GeneratedCode(
            code="class Foo(: pass",  # Invalid syntax
            name="Foo",
            num_components=1,
            component_names=["test"]
        )
        model, error = builder.build(generated)
        assert model is None
        assert "Syntax error" in error

    def test_build_missing_class(self, builder):
        """Test building code without the expected class."""
        generated = GeneratedCode(
            code="class WrongName:\n    pass",
            name="ExpectedName",
            num_components=1,
            component_names=["test"]
        )
        model, error = builder.build(generated)
        assert model is None
        assert "not found" in error

    def test_build_valid_code(self, builder):
        """Test building valid PyTorch code."""
        import torch.nn as nn

        code = '''
import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self, d_model=128, vocab_size=1000, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x, **kwargs):
        return self.linear(x)
'''
        generated = GeneratedCode(
            code=code,
            name="TestModel",
            num_components=1,
            component_names=["test"]
        )
        model, error = builder.build(generated)
        assert error is None
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_count_parameters(self, builder):
        """Test parameter counting."""
        import torch.nn as nn

        model = nn.Linear(10, 10)
        count = builder.count_parameters(model)
        # 10*10 weights + 10 bias = 110
        assert count == 110


class TestSyntheticDataset:
    """Test SyntheticDataset class."""

    def test_dataset_length(self):
        """Test dataset has correct length."""
        from arcfusion.validator import SyntheticDataset
        dataset = SyntheticDataset(vocab_size=100, seq_len=32, num_samples=500)
        assert len(dataset) == 500

    def test_dataset_item_shape(self):
        """Test dataset items have correct shape."""
        from arcfusion.validator import SyntheticDataset
        dataset = SyntheticDataset(vocab_size=100, seq_len=32, num_samples=100)
        inputs, targets = dataset[0]
        assert inputs.shape == (32,)
        assert targets.shape == (32,)

    def test_dataset_vocab_range(self):
        """Test dataset values are within vocab range."""
        from arcfusion.validator import SyntheticDataset
        dataset = SyntheticDataset(vocab_size=100, seq_len=32, num_samples=100)
        inputs, targets = dataset[0]
        assert inputs.max() < 100
        assert targets.max() < 100


class TestBenchmarkRunner:
    """Test BenchmarkRunner class."""

    def test_compute_perplexity(self):
        """Test perplexity computation from loss."""
        import math
        from arcfusion.validator import BenchmarkRunner

        runner = BenchmarkRunner()

        # Loss of 0 should give perplexity of 1
        assert runner.compute_perplexity(0) == pytest.approx(1.0)

        # Loss of 1 should give perplexity of e
        assert runner.compute_perplexity(1) == pytest.approx(math.e, rel=0.01)

        # High loss should be clamped
        ppl = runner.compute_perplexity(100)
        assert ppl < float('inf')


class TestValidationPipeline:
    """Test full validation pipeline."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        from arcfusion.validator import ValidationPipeline, ModelConfig, TrainingConfig
        from arcfusion import ArcFusionDB

        db_path = tmp_path / "test.db"
        db = ArcFusionDB(str(db_path))

        # Use very small config for fast tests
        model_config = ModelConfig(d_model=32, vocab_size=100, max_seq_len=16)
        training_config = TrainingConfig(max_steps=5, batch_size=2)

        return ValidationPipeline(db, model_config, training_config)

    def test_validate_invalid_code(self, pipeline):
        """Test validation of invalid code fails gracefully."""
        generated = GeneratedCode(
            code="invalid python code {{{{",
            name="BadModel",
            num_components=1,
            component_names=["test"]
        )
        result = pipeline.validate(generated, verbose=False)
        assert not result.success
        assert result.build_error is not None

    def test_validate_simple_model(self, pipeline):
        """Test validation of a simple valid model."""
        code = '''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self, d_model=32, vocab_size=100, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x, **kwargs):
        return self.linear(x)
'''
        generated = GeneratedCode(
            code=code,
            name="SimpleModel",
            num_components=1,
            component_names=["linear"]
        )
        result = pipeline.validate(generated, verbose=False)

        # Should succeed
        assert result.success
        assert result.build_error is None
        assert result.train_error is None
        assert result.num_parameters > 0
        assert result.training_steps > 0
        assert result.final_loss < float('inf')
        assert result.perplexity < float('inf')


# Tests that don't require torch
class TestValidatorImports:
    """Test that validator handles missing torch gracefully."""

    def test_has_validator_flag(self):
        """Test HAS_VALIDATOR flag is set correctly."""
        from arcfusion import HAS_VALIDATOR
        # If we got here, torch is available
        assert HAS_VALIDATOR is True
