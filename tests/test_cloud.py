"""Tests for cloud training module."""

import pytest


class TestCloudConfig:
    """Test CloudConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from arcfusion.cloud import CloudConfig

        config = CloudConfig()
        assert config.gpu_type == "T4"
        assert config.timeout_seconds == 600
        assert config.max_steps == 2000
        assert config.batch_size == 16
        assert config.d_model == 256
        assert config.vocab_size == 8000
        assert config.learning_rate == 1e-4

    def test_custom_values(self):
        """Test custom configuration values."""
        from arcfusion.cloud import CloudConfig

        config = CloudConfig(
            gpu_type="A100",
            max_steps=1000,
            batch_size=32,
            d_model=512,
        )
        assert config.gpu_type == "A100"
        assert config.max_steps == 1000
        assert config.batch_size == 32
        assert config.d_model == 512


class TestCloudResult:
    """Test CloudResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        from arcfusion.cloud import CloudResult

        result = CloudResult(success=False)
        assert not result.success
        assert result.final_loss == float('inf')
        assert result.perplexity == float('inf')
        assert result.steps_completed == 0
        assert result.error is None
        assert result.loss_history == []

    def test_successful_result(self):
        """Test successful result with values."""
        from arcfusion.cloud import CloudResult

        result = CloudResult(
            success=True,
            final_loss=2.5,
            perplexity=12.18,
            steps_completed=500,
            training_time_seconds=45.2,
            num_parameters=1_000_000,
        )
        assert result.success
        assert result.final_loss == 2.5
        assert result.perplexity == 12.18
        assert result.steps_completed == 500
        assert result.num_parameters == 1_000_000

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from arcfusion.cloud import CloudResult

        result = CloudResult(
            success=True,
            final_loss=2.5,
            perplexity=12.18,
            steps_completed=500,
            loss_history=[{'step': 0, 'loss': 10.0}, {'step': 100, 'loss': 5.0}],
        )
        d = result.to_dict()

        assert d['success'] is True
        assert d['final_loss'] == 2.5
        assert d['perplexity'] == 12.18
        assert len(d['loss_history']) == 2

    def test_from_dict(self):
        """Test creation from dictionary."""
        from arcfusion.cloud import CloudResult

        d = {
            'success': True,
            'final_loss': 2.5,
            'perplexity': 12.18,
            'steps_completed': 500,
            'training_time_seconds': 45.2,
            'num_parameters': 1_000_000,
            'error': None,
            'loss_history': [],
        }
        result = CloudResult.from_dict(d)

        assert result.success
        assert result.final_loss == 2.5
        assert result.steps_completed == 500

    def test_roundtrip(self):
        """Test dict roundtrip preserves data."""
        from arcfusion.cloud import CloudResult

        original = CloudResult(
            success=True,
            final_loss=2.5,
            perplexity=12.18,
            steps_completed=500,
            training_time_seconds=45.2,
            num_parameters=1_000_000,
            loss_history=[{'step': 0, 'loss': 10.0}],
        )
        roundtripped = CloudResult.from_dict(original.to_dict())

        assert roundtripped.success == original.success
        assert roundtripped.final_loss == original.final_loss
        assert roundtripped.perplexity == original.perplexity
        assert roundtripped.steps_completed == original.steps_completed
        assert roundtripped.loss_history == original.loss_history


class TestCloudModuleImport:
    """Test cloud module import behavior."""

    def test_has_modal_flag(self):
        """Test HAS_MODAL flag is set."""
        from arcfusion.cloud import HAS_MODAL

        # HAS_MODAL is True if modal is installed, False otherwise
        assert isinstance(HAS_MODAL, bool)

    def test_constants_defined(self):
        """Test module constants are defined."""
        from arcfusion.cloud import (
            APP_NAME,
            GPU_TYPE,
            TIMEOUT_SECONDS,
            CLOUD_DEFAULT_STEPS,
        )

        assert APP_NAME == "arcfusion-trainer"
        assert GPU_TYPE == "T4"
        assert TIMEOUT_SECONDS == 600
        assert CLOUD_DEFAULT_STEPS == 2000


class TestCloudTrainerInit:
    """Test CloudTrainer initialization."""

    def test_init_without_modal(self):
        """Test that CloudTrainer requires modal."""
        from arcfusion.cloud import HAS_MODAL

        if not HAS_MODAL:
            from arcfusion.cloud import CloudTrainer

            with pytest.raises(ImportError, match="Modal not installed"):
                CloudTrainer()

    def test_init_with_config(self):
        """Test CloudTrainer accepts config."""
        from arcfusion.cloud import HAS_MODAL, CloudConfig

        if HAS_MODAL:
            from arcfusion.cloud import CloudTrainer

            config = CloudConfig(max_steps=100)
            trainer = CloudTrainer(config=config)
            assert trainer.config.max_steps == 100


class TestPackageExports:
    """Test package-level exports."""

    def test_cloud_exports(self):
        """Test cloud classes are exported from package."""
        import arcfusion

        # These should be available (may be None if modal not installed)
        assert hasattr(arcfusion, 'CloudTrainer')
        assert hasattr(arcfusion, 'CloudConfig')
        assert hasattr(arcfusion, 'CloudResult')
        assert hasattr(arcfusion, 'HAS_CLOUD')

    def test_has_cloud_flag(self):
        """Test HAS_CLOUD flag matches modal availability."""
        import arcfusion
        from arcfusion.cloud import HAS_MODAL

        assert arcfusion.HAS_CLOUD == HAS_MODAL
