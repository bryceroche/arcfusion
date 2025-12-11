"""
Tests for LLM-powered paper analyzer.
Tests dataclass parsing and database integration without requiring API key.
"""

import pytest
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from arcfusion.analyzer import AnalyzedComponent, AnalysisResult
from arcfusion.db import ArcFusionDB, Component, Engine


class TestAnalyzedComponent:
    """Test AnalyzedComponent dataclass."""

    def test_create_analyzed_component(self):
        comp = AnalyzedComponent(
            name="TestComponent",
            description="A test component",
            category="attention",
            interface_in={"shape": "[batch, seq, d]", "dtype": "float32"},
            interface_out={"shape": "[batch, seq, d]", "dtype": "float32"},
            hyperparameters={"dim": 512},
            time_complexity="O(n)",
            space_complexity="O(n)",
            flops_formula="2*n*d",
            math_operations=["matmul", "relu"],
            is_parallelizable=True,
            is_causal=False,
            innovation="Novel approach to X",
            builds_on=["BaseComponent"],
            confidence=0.9,
            code_sketch="def forward(self, x): return x",
        )
        assert comp.name == "TestComponent"
        assert comp.category == "attention"
        assert comp.confidence == 0.9
        assert "matmul" in comp.math_operations

    def test_analyzed_component_defaults(self):
        comp = AnalyzedComponent(
            name="Minimal",
            description="Minimal component",
            category="layer",
            interface_in={},
            interface_out={},
            hyperparameters={},
            time_complexity="",
            space_complexity="",
            flops_formula="",
            math_operations=[],
            is_parallelizable=True,
            is_causal=False,
            innovation="",
            builds_on=[],
            confidence=0.5,
            code_sketch="",
        )
        assert comp.name == "Minimal"
        assert comp.category == "layer"
        assert comp.math_operations == []


class TestAnalysisResult:
    """Test AnalysisResult dataclass."""

    def test_create_analysis_result(self):
        comp = AnalyzedComponent(
            name="TestComp",
            description="desc",
            category="structure",
            interface_in={},
            interface_out={},
            hyperparameters={},
            time_complexity="O(n)",
            space_complexity="O(n)",
            flops_formula="",
            math_operations=[],
            is_parallelizable=True,
            is_causal=False,
            innovation="innovation",
            builds_on=[],
            confidence=0.8,
            code_sketch="",
        )
        result = AnalysisResult(
            paper_title="Test Paper",
            paper_id="1234.56789",
            architecture_name="TestArch",
            architecture_description="Test architecture description",
            novel_components=[comp],
            component_relationships=[("A", "B", 0.9, "reason")],
            key_innovations=["innovation 1"],
            limitations=["limitation 1"],
        )
        assert result.paper_title == "Test Paper"
        assert len(result.novel_components) == 1
        assert result.novel_components[0].name == "TestComp"

    def test_empty_analysis_result(self):
        result = AnalysisResult(
            paper_title="Empty Paper",
            paper_id="0000.00000",
            architecture_name="",
            architecture_description="",
            novel_components=[],
            component_relationships=[],
            key_innovations=[],
            limitations=[],
        )
        assert len(result.novel_components) == 0


class TestAnalyzerIntegration:
    """Test analyzer database integration (without API calls)."""

    @pytest.fixture
    def db(self):
        """Create a temporary test database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        db = ArcFusionDB(db_path)
        yield db
        db.close()
        os.unlink(db_path)

    def test_convert_analyzed_to_db_component(self, db):
        """Test converting AnalyzedComponent to DB Component."""
        analyzed = AnalyzedComponent(
            name="SelectiveSSM",
            description="Selective state space model",
            category="attention",
            interface_in={"shape": "[b, n, d]", "dtype": "float32"},
            interface_out={"shape": "[b, n, d]", "dtype": "float32"},
            hyperparameters={"d_state": 16, "d_conv": 4},
            time_complexity="O(n*d*s)",
            space_complexity="O(n*s)",
            flops_formula="6*n*d*s",
            math_operations=["linear", "conv1d", "ssm_scan"],
            is_parallelizable=True,
            is_causal=True,
            innovation="Input-dependent parameters",
            builds_on=["S4"],
            confidence=0.95,
            code_sketch="def forward(self, x): return y",
        )

        # Convert to DB component (simulating what analyze_and_ingest does)
        comp = Component(
            name=analyzed.name,
            description=analyzed.description,
            interface_in=analyzed.interface_in,
            interface_out=analyzed.interface_out,
            code=analyzed.code_sketch,
            usefulness_score=analyzed.confidence,
            source_paper_id="2312.00752",
            introduced_year=2023,
            hyperparameters=analyzed.hyperparameters,
            time_complexity=analyzed.time_complexity,
            space_complexity=analyzed.space_complexity,
            flops_formula=analyzed.flops_formula,
            is_parallelizable=analyzed.is_parallelizable,
            is_causal=analyzed.is_causal,
            math_operations=analyzed.math_operations,
        )

        db.add_component(comp)

        # Verify retrieval
        retrieved = db.get_component(comp.component_id)
        assert retrieved.name == "SelectiveSSM"
        assert retrieved.time_complexity == "O(n*d*s)"
        assert retrieved.is_causal is True
        assert "ssm_scan" in retrieved.math_operations
        assert retrieved.hyperparameters["d_state"] == 16

    def test_deduplication_logic(self, db):
        """Test that duplicate components are detected."""
        # Add initial component
        comp1 = Component(
            name="Attention",
            description="Standard attention",
            interface_in={},
            interface_out={},
            usefulness_score=0.8,
        )
        db.add_component(comp1)

        # Simulate checking for duplicates (case-insensitive)
        existing = {c.name.lower(): c for c in db.find_components()}

        # These should be detected as duplicates
        assert "attention" in existing
        assert "ATTENTION".lower() in existing

        # This should not exist
        assert "flashattention" not in existing

    def test_confidence_threshold(self, db):
        """Test that low confidence components are filtered."""
        min_confidence = 0.7

        components = [
            AnalyzedComponent(
                name="HighConfComp",
                description="High confidence",
                category="layer",
                interface_in={},
                interface_out={},
                hyperparameters={},
                time_complexity="",
                space_complexity="",
                flops_formula="",
                math_operations=[],
                is_parallelizable=True,
                is_causal=False,
                innovation="",
                builds_on=[],
                confidence=0.9,
                code_sketch="",
            ),
            AnalyzedComponent(
                name="LowConfComp",
                description="Low confidence",
                category="layer",
                interface_in={},
                interface_out={},
                hyperparameters={},
                time_complexity="",
                space_complexity="",
                flops_formula="",
                math_operations=[],
                is_parallelizable=True,
                is_causal=False,
                innovation="",
                builds_on=[],
                confidence=0.5,
                code_sketch="",
            ),
        ]

        # Simulate filtering
        filtered = [c for c in components if c.confidence >= min_confidence]
        assert len(filtered) == 1
        assert filtered[0].name == "HighConfComp"


class TestJSONParsing:
    """Test JSON response parsing from LLM."""

    def test_parse_full_response(self):
        """Test parsing a complete LLM JSON response."""
        json_data = {
            "architecture_name": "Mamba",
            "architecture_description": "Selective SSM architecture",
            "novel_components": [
                {
                    "name": "SelectiveSSM",
                    "category": "attention",
                    "description": "SSM with input-dependent params",
                    "interface_in": {"shape": "[b, n, d]"},
                    "interface_out": {"shape": "[b, n, d]"},
                    "hyperparameters": {"d_state": 16},
                    "time_complexity": "O(n)",
                    "space_complexity": "O(n)",
                    "flops_formula": "6nd",
                    "math_operations": ["ssm_scan"],
                    "is_parallelizable": True,
                    "is_causal": True,
                    "innovation": "Selective parameters",
                    "builds_on": ["S4"],
                    "confidence": 0.95,
                    "code_sketch": "pass"
                }
            ],
            "component_relationships": [
                ["Comp1", "Comp2", 0.9, "reason"]
            ],
            "key_innovations": ["innovation"],
            "limitations": ["limitation"]
        }

        # Parse components
        components = []
        for comp_data in json_data["novel_components"]:
            comp = AnalyzedComponent(
                name=comp_data.get("name", "Unknown"),
                description=comp_data.get("description", ""),
                category=comp_data.get("category", "layer"),
                interface_in=comp_data.get("interface_in", {}),
                interface_out=comp_data.get("interface_out", {}),
                hyperparameters=comp_data.get("hyperparameters", {}),
                time_complexity=comp_data.get("time_complexity", ""),
                space_complexity=comp_data.get("space_complexity", ""),
                flops_formula=comp_data.get("flops_formula", ""),
                math_operations=comp_data.get("math_operations", []),
                is_parallelizable=comp_data.get("is_parallelizable", True),
                is_causal=comp_data.get("is_causal", False),
                innovation=comp_data.get("innovation", ""),
                builds_on=comp_data.get("builds_on", []),
                confidence=comp_data.get("confidence", 0.5),
                code_sketch=comp_data.get("code_sketch", ""),
            )
            components.append(comp)

        result = AnalysisResult(
            paper_title="Mamba Paper",
            paper_id="2312.00752",
            architecture_name=json_data["architecture_name"],
            architecture_description=json_data["architecture_description"],
            novel_components=components,
            component_relationships=json_data["component_relationships"],
            key_innovations=json_data["key_innovations"],
            limitations=json_data["limitations"],
        )

        assert result.architecture_name == "Mamba"
        assert len(result.novel_components) == 1
        assert result.novel_components[0].confidence == 0.95
        assert result.novel_components[0].category == "attention"

    def test_handle_missing_fields(self):
        """Test graceful handling of missing optional fields."""
        minimal_data = {
            "name": "MinimalComponent",
        }

        comp = AnalyzedComponent(
            name=minimal_data.get("name", "Unknown"),
            description=minimal_data.get("description", ""),
            category=minimal_data.get("category", "layer"),
            interface_in=minimal_data.get("interface_in", {}),
            interface_out=minimal_data.get("interface_out", {}),
            hyperparameters=minimal_data.get("hyperparameters", {}),
            time_complexity=minimal_data.get("time_complexity", ""),
            space_complexity=minimal_data.get("space_complexity", ""),
            flops_formula=minimal_data.get("flops_formula", ""),
            math_operations=minimal_data.get("math_operations", []),
            is_parallelizable=minimal_data.get("is_parallelizable", True),
            is_causal=minimal_data.get("is_causal", False),
            innovation=minimal_data.get("innovation", ""),
            builds_on=minimal_data.get("builds_on", []),
            confidence=minimal_data.get("confidence", 0.5),
            code_sketch=minimal_data.get("code_sketch", ""),
        )

        assert comp.name == "MinimalComponent"
        assert comp.description == ""
        assert comp.category == "layer"
        assert comp.confidence == 0.5


class TestRelationshipValidation:
    """Test relationship validation logic."""

    def test_valid_relationship_score(self):
        """Test that valid scores are accepted."""
        valid_scores = [0.0, 0.5, 1.0, 0.75]
        for score in valid_scores:
            assert isinstance(score, (int, float))
            assert 0 <= score <= 1

    def test_invalid_relationship_score_detection(self):
        """Test that invalid scores are detected."""
        invalid_scores = [-0.1, 1.5, "high", None]
        for score in invalid_scores:
            is_valid = isinstance(score, (int, float)) and 0 <= score <= 1 if isinstance(score, (int, float)) else False
            assert not is_valid

    def test_relationship_requires_both_components(self):
        """Test that relationships are only created when both components exist."""
        existing = {"comp1": "id1", "comp2": "id2"}

        # Both exist - should succeed
        rel1 = ("comp1", "comp2", 0.9, "reason")
        comp1 = existing.get(rel1[0].lower())
        comp2 = existing.get(rel1[1].lower())
        assert comp1 and comp2

        # One missing - should fail
        rel2 = ("comp1", "missing", 0.9, "reason")
        comp1 = existing.get(rel2[0].lower())
        comp2 = existing.get(rel2[1].lower())
        assert not (comp1 and comp2)

        # Both missing - should fail
        rel3 = ("missing1", "missing2", 0.9, "reason")
        comp1 = existing.get(rel3[0].lower())
        comp2 = existing.get(rel3[1].lower())
        assert not (comp1 and comp2)


class TestComponentOrdering:
    """Test component ordering extraction and handling."""

    def test_analyzed_component_has_position(self):
        """Test that AnalyzedComponent has position field."""
        comp = AnalyzedComponent(
            name="TestComp",
            description="Test",
            category="layer",
            interface_in={},
            interface_out={},
            hyperparameters={},
            time_complexity="",
            space_complexity="",
            flops_formula="",
            math_operations=[],
            is_parallelizable=True,
            is_causal=False,
            innovation="",
            builds_on=[],
            confidence=0.8,
            code_sketch="",
            position=3,
        )
        assert comp.position == 3

    def test_position_defaults_to_zero(self):
        """Test that position defaults to 0 when not specified."""
        comp = AnalyzedComponent(
            name="DefaultPos",
            description="Test",
            category="layer",
            interface_in={},
            interface_out={},
            hyperparameters={},
            time_complexity="",
            space_complexity="",
            flops_formula="",
            math_operations=[],
            is_parallelizable=True,
            is_causal=False,
            innovation="",
            builds_on=[],
            confidence=0.8,
            code_sketch="",
        )
        assert comp.position == 0

    def test_components_sorted_by_position(self):
        """Test that components are sorted by position."""
        # Create components in wrong order
        comp_attention = AnalyzedComponent(
            name="Attention", description="", category="attention",
            interface_in={}, interface_out={}, hyperparameters={},
            time_complexity="", space_complexity="", flops_formula="",
            math_operations=[], is_parallelizable=True, is_causal=False,
            innovation="", builds_on=[], confidence=0.9, code_sketch="",
            position=2,
        )
        comp_embedding = AnalyzedComponent(
            name="Embedding", description="", category="position",
            interface_in={}, interface_out={}, hyperparameters={},
            time_complexity="", space_complexity="", flops_formula="",
            math_operations=[], is_parallelizable=True, is_causal=False,
            innovation="", builds_on=[], confidence=0.9, code_sketch="",
            position=0,
        )
        comp_output = AnalyzedComponent(
            name="Output", description="", category="output",
            interface_in={}, interface_out={}, hyperparameters={},
            time_complexity="", space_complexity="", flops_formula="",
            math_operations=[], is_parallelizable=True, is_causal=False,
            innovation="", builds_on=[], confidence=0.9, code_sketch="",
            position=4,
        )
        comp_ffn = AnalyzedComponent(
            name="FFN", description="", category="layer",
            interface_in={}, interface_out={}, hyperparameters={},
            time_complexity="", space_complexity="", flops_formula="",
            math_operations=[], is_parallelizable=True, is_causal=False,
            innovation="", builds_on=[], confidence=0.9, code_sketch="",
            position=3,
        )
        comp_norm = AnalyzedComponent(
            name="LayerNorm", description="", category="layer",
            interface_in={}, interface_out={}, hyperparameters={},
            time_complexity="", space_complexity="", flops_formula="",
            math_operations=[], is_parallelizable=True, is_causal=False,
            innovation="", builds_on=[], confidence=0.9, code_sketch="",
            position=1,
        )

        # Unsorted list
        components = [comp_attention, comp_embedding, comp_output, comp_ffn, comp_norm]

        # Sort by position
        sorted_components = sorted(components, key=lambda c: c.position)

        assert sorted_components[0].name == "Embedding"
        assert sorted_components[1].name == "LayerNorm"
        assert sorted_components[2].name == "Attention"
        assert sorted_components[3].name == "FFN"
        assert sorted_components[4].name == "Output"

    def test_parse_position_from_json(self):
        """Test parsing position from JSON response."""
        json_data = {
            "novel_components": [
                {"name": "Output", "position": 3, "category": "output"},
                {"name": "Embedding", "position": 0, "category": "position"},
                {"name": "Attention", "position": 1, "category": "attention"},
                {"name": "FFN", "position": 2, "category": "layer"},
            ]
        }

        components = []
        for i, comp_data in enumerate(json_data["novel_components"]):
            comp = AnalyzedComponent(
                name=comp_data.get("name", "Unknown"),
                description="",
                category=comp_data.get("category", "layer"),
                interface_in={},
                interface_out={},
                hyperparameters={},
                time_complexity="",
                space_complexity="",
                flops_formula="",
                math_operations=[],
                is_parallelizable=True,
                is_causal=False,
                innovation="",
                builds_on=[],
                confidence=0.5,
                code_sketch="",
                position=comp_data.get("position", i),
            )
            components.append(comp)

        # Sort by position
        components.sort(key=lambda c: c.position)

        # Verify order matches architectural flow
        assert [c.name for c in components] == ["Embedding", "Attention", "FFN", "Output"]

    def test_position_fallback_to_index(self):
        """Test that position falls back to index when not provided."""
        json_data = {
            "novel_components": [
                {"name": "First"},
                {"name": "Second"},
                {"name": "Third"},
            ]
        }

        components = []
        for i, comp_data in enumerate(json_data["novel_components"]):
            comp = AnalyzedComponent(
                name=comp_data.get("name", "Unknown"),
                description="",
                category="layer",
                interface_in={},
                interface_out={},
                hyperparameters={},
                time_complexity="",
                space_complexity="",
                flops_formula="",
                math_operations=[],
                is_parallelizable=True,
                is_causal=False,
                innovation="",
                builds_on=[],
                confidence=0.5,
                code_sketch="",
                position=comp_data.get("position", i),  # Fallback to index
            )
            components.append(comp)

        assert components[0].position == 0
        assert components[1].position == 1
        assert components[2].position == 2
