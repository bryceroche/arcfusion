"""
Tests for code generation module.
"""

import pytest
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from arcfusion.codegen import (
    CodeGenerator,
    GeneratedCode,
    sanitize_class_name,
    sanitize_var_name,
    extract_forward_body,
    generate_component_class,
    generate_full_module,
)
from arcfusion.db import ArcFusionDB, Component


class TestSanitization:
    """Test name sanitization functions."""

    def test_sanitize_class_name_basic(self):
        assert sanitize_class_name("MultiHeadAttention") == "MultiHeadAttention"
        assert sanitize_class_name("Layer Normalization") == "LayerNormalization"

    def test_sanitize_class_name_with_parens(self):
        assert sanitize_class_name("RMSNorm (Root Mean Square)") == "RMSNorm"
        assert sanitize_class_name("BERT (Bidirectional)") == "BERT"

    def test_sanitize_class_name_special_chars(self):
        assert sanitize_class_name("Position-wise FFN") == "PositionwiseFFN"
        assert sanitize_class_name("KV-Cache") == "KVCache"

    def test_sanitize_class_name_starts_with_digit(self):
        assert sanitize_class_name("2DAttention") == "M2DAttention"

    def test_sanitize_var_name_basic(self):
        assert sanitize_var_name("MultiHeadAttention") == "multi_head_attention"
        assert sanitize_var_name("LayerNorm") == "layer_norm"

    def test_sanitize_var_name_with_parens(self):
        result = sanitize_var_name("BERT (Bidirectional)")
        # All caps get split per letter in snake_case
        assert result.startswith("b")
        assert "_" in result

    def test_sanitize_var_name_special_chars(self):
        result = sanitize_var_name("Position-wise FFN")
        assert "position" in result
        assert "wise" in result


class TestExtractForwardBody:
    """Test forward method extraction."""

    def test_extract_simple_forward(self):
        code = """
def forward(self, x):
    return x + 1
"""
        result = extract_forward_body(code)
        assert "return x + 1" in result

    def test_extract_multiline_forward(self):
        code = """
def forward(self, x, mask):
    x = self.attention(x, mask)
    x = self.norm(x)
    return x
"""
        result = extract_forward_body(code)
        assert "self.attention" in result
        assert "self.norm" in result

    def test_extract_no_forward(self):
        code = """
def __init__(self):
    pass
"""
        result = extract_forward_body(code)
        assert result == "return x"

    def test_extract_empty_code(self):
        assert extract_forward_body("") == "return x"
        assert extract_forward_body(None) == "return x"


class TestGenerateComponentClass:
    """Test component class generation."""

    def test_generate_attention_component(self):
        comp = Component(
            name="Multi-Head Attention",
            description="Attention mechanism",
            interface_in={"shape": "[batch, seq, d]"},
            interface_out={"shape": "[batch, seq, d]"},
            code="def forward(self, x): return x",
        )
        result = generate_component_class(comp, 0)

        assert "class MultiHeadAttention(nn.Module):" in result
        assert "def __init__" in result
        assert "def forward" in result
        assert "super().__init__()" in result

    def test_generate_normalization_component(self):
        comp = Component(
            name="Layer Normalization",
            description="Normalize layer outputs",
            interface_in={"shape": "[batch, seq, d]"},
            interface_out={"shape": "[batch, seq, d]"},
        )
        result = generate_component_class(comp, 0)

        assert "class LayerNormalization(nn.Module):" in result
        assert "nn.LayerNorm" in result

    def test_generate_output_component(self):
        comp = Component(
            name="Output Projection",
            description="Project to vocabulary",
            interface_in={"shape": "[batch, seq, d]"},
            interface_out={"shape": "[batch, seq, vocab]"},
        )
        result = generate_component_class(comp, 0)

        assert "class OutputProjection(nn.Module):" in result
        assert "vocab_size" in result


class TestGeneratedCode:
    """Test GeneratedCode dataclass."""

    def test_validate_syntax_valid(self):
        code = """
import torch
class Foo:
    pass
"""
        gc = GeneratedCode(code=code, name="Foo", num_components=1, component_names=["Foo"])
        valid, error = gc.validate_syntax()
        assert valid is True
        assert error is None

    def test_validate_syntax_invalid(self):
        code = """
class Foo
    pass
"""
        gc = GeneratedCode(code=code, name="Foo", num_components=1, component_names=["Foo"])
        valid, error = gc.validate_syntax()
        assert valid is False
        assert error is not None

    def test_save_file(self):
        code = "# test code\nprint('hello')"
        gc = GeneratedCode(code=code, name="Test", num_components=0, component_names=[])

        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
            path = f.name

        try:
            gc.save(path)
            with open(path) as f:
                content = f.read()
            assert content == code
        finally:
            os.unlink(path)

    def test_save_validates_by_default(self):
        """Test that save() validates syntax by default."""
        invalid_code = "class Foo\n    pass"  # Missing colon
        gc = GeneratedCode(code=invalid_code, name="Invalid", num_components=0, component_names=[])

        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
            path = f.name

        try:
            with pytest.raises(ValueError, match="Cannot save invalid Python code"):
                gc.save(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_can_skip_validation(self):
        """Test that validation can be skipped."""
        invalid_code = "class Foo\n    pass"  # Missing colon
        gc = GeneratedCode(code=invalid_code, name="Invalid", num_components=0, component_names=[])

        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
            path = f.name

        try:
            # Should succeed with validate=False
            result = gc.save(path, validate=False)
            assert result is True
            with open(path) as f:
                content = f.read()
            assert content == invalid_code
        finally:
            os.unlink(path)


class TestCodeGeneratorIntegration:
    """Integration tests with database."""

    @pytest.fixture
    def db(self):
        """Create a temporary test database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        db = ArcFusionDB(db_path)

        # Add test components
        comp1 = Component(
            name="Test Attention",
            description="Test attention component",
            interface_in={"shape": "[b, n, d]"},
            interface_out={"shape": "[b, n, d]"},
            code="def forward(self, x): return x",
            usefulness_score=0.9,
        )
        comp2 = Component(
            name="Test Norm",
            description="Test normalization",
            interface_in={"shape": "[b, n, d]"},
            interface_out={"shape": "[b, n, d]"},
            code="def forward(self, x): return self.norm(x)",
            usefulness_score=0.8,
        )
        db.add_component(comp1)
        db.add_component(comp2)

        yield db
        db.close()
        os.unlink(db_path)

    def test_generate_from_components(self, db):
        gen = CodeGenerator(db)
        components = db.find_components()

        result = gen.generate(components, name="TestArch")

        assert result.name == "TestArch"
        assert result.num_components == 2
        # Component names are original names, not sanitized
        assert any("Attention" in name or "Norm" in name for name in result.component_names)

        valid, error = result.validate_syntax()
        assert valid, f"Syntax error: {error}"

    def test_generate_full_module(self, db):
        components = db.find_components()
        code = generate_full_module(components, name="FullModule")

        assert "import torch" in code
        assert "import torch.nn as nn" in code
        assert "class FullModule(nn.Module):" in code
        assert "if __name__" in code

        # Verify it compiles
        compile(code, '<test>', 'exec')


class TestGenerateFullModule:
    """Test full module generation."""

    def test_includes_imports(self):
        comp = Component(
            name="Simple",
            description="Simple component",
            interface_in={},
            interface_out={},
        )
        code = generate_full_module([comp], include_imports=True)

        assert "import math" in code
        assert "import torch" in code
        assert "import torch.nn as nn" in code

    def test_excludes_imports(self):
        comp = Component(
            name="Simple",
            description="Simple component",
            interface_in={},
            interface_out={},
        )
        code = generate_full_module([comp], include_imports=False)

        assert "import torch" not in code

    def test_includes_example(self):
        comp = Component(
            name="Simple",
            description="Simple component",
            interface_in={},
            interface_out={},
        )
        code = generate_full_module([comp], include_example=True)

        assert 'if __name__ == "__main__":' in code
        assert "torch.randn" in code

    def test_excludes_example(self):
        comp = Component(
            name="Simple",
            description="Simple component",
            interface_in={},
            interface_out={},
        )
        code = generate_full_module([comp], include_example=False)

        assert 'if __name__ == "__main__":' not in code
