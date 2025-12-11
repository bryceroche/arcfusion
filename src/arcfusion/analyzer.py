"""
LLM-Powered Paper Analyzer - Deep component extraction using Claude.

Instead of keyword matching, uses LLM to understand papers and extract:
- Specific novel components (not generic categories)
- Exact interfaces and hyperparameters
- Complexity analysis
- What makes it different from existing components
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from .db import ArcFusionDB, Component, Engine, ComponentRelationship


@dataclass
class AnalyzedComponent:
    """Result of LLM component analysis"""
    name: str
    description: str
    category: str  # attention, structure, layer, position, training, efficiency, output
    interface_in: dict
    interface_out: dict
    hyperparameters: dict
    time_complexity: str
    space_complexity: str
    flops_formula: str
    math_operations: list
    is_parallelizable: bool
    is_causal: bool
    innovation: str  # What makes this novel
    builds_on: list  # Existing components it extends/replaces
    confidence: float  # How confident the LLM is (0-1)
    code_sketch: str  # PyTorch pseudocode
    position: int = 0  # Order in architecture (0=first, higher=later in forward pass)


@dataclass
class AnalysisResult:
    """Full paper analysis result"""
    paper_title: str
    paper_id: str
    architecture_name: str
    architecture_description: str
    novel_components: list[AnalyzedComponent]
    component_relationships: list[tuple[str, str, float, str]]  # (comp1, comp2, score, reason)
    key_innovations: list[str]
    limitations: list[str]


ANALYSIS_PROMPT = '''You are an ML architecture expert with comprehensive knowledge of research papers.
Given the paper information below, extract ALL architectural components introduced or specifically defined in this paper.

Paper Title: {title}
Paper ID: {paper_id}

Abstract:
{content}

IMPORTANT: You have deep knowledge of this paper beyond just the abstract. Use your full understanding of "{title}" to extract EVERY architectural component, including:

1. **Structural Components**: encoder blocks, decoder blocks, full transformer blocks, complete layer stacks
2. **Attention Variants**:
   - Self-attention (queries, keys, values from same source)
   - Cross-attention / Encoder-Decoder attention (queries from decoder, keys/values from encoder)
   - Masked/Causal attention (future positions masked)
   - Any novel attention mechanisms (linear, sparse, flash, etc.)
3. **Layer Components**: feed-forward networks, normalization layers, residual connections, gating mechanisms
4. **Positional Components**: sinusoidal encodings, learned embeddings, rotary embeddings, ALiBi
5. **Training Components**: learning rate schedules, optimizer configs, regularization, loss functions
6. **Efficiency Components**: KV-cache, quantization, pruning, distillation techniques
7. **Output Components**: output projections, vocabulary mappings, decoding strategies

For EACH component, provide:
- The specific name as used in or derived from this paper
- Category (one of: attention, structure, layer, position, training, efficiency, output)
- **Position in architecture** (0=first component in forward pass, increasing numbers for later components)
- Detailed description of what it does
- Input/output interfaces with shapes
- All hyperparameters with the values used in the paper
- Time and space complexity
- Core math operations
- Whether parallelizable and/or causal
- What makes it novel or how it differs from prior work
- PyTorch code sketch

IMPORTANT: Component ORDER matters! List components in the order they appear in the forward pass:
- For Transformer: Embedding(0) → PositionalEncoding(1) → Attention(2) → LayerNorm(3) → FFN(4) → Output(5)
- The "position" field should reflect where data flows through this component

Include components even if they build on prior work - what matters is how THIS paper defines/uses them.

Respond in this exact JSON format:
{{
    "architecture_name": "Name of the overall architecture",
    "architecture_description": "Comprehensive description of the full architecture",
    "novel_components": [
        {{
            "name": "ComponentName",
            "category": "attention|structure|layer|position|training|efficiency|output",
            "position": 0,
            "description": "Detailed description",
            "interface_in": {{"shape": "[batch, seq_len, d_model]", "dtype": "float32"}},
            "interface_out": {{"shape": "[batch, seq_len, d_model]", "dtype": "float32"}},
            "hyperparameters": {{"param": "value"}},
            "time_complexity": "O(...)",
            "space_complexity": "O(...)",
            "flops_formula": "formula",
            "math_operations": ["op1", "op2"],
            "is_parallelizable": true,
            "is_causal": false,
            "innovation": "What's novel or specific to this paper",
            "builds_on": ["prior components"],
            "confidence": 0.9,
            "code_sketch": "def forward(self, x): ..."
        }}
    ],
    "component_relationships": [
        ["Component1", "Component2", 0.95, "Why they work together"]
    ],
    "key_innovations": ["Innovation 1", "Innovation 2"],
    "limitations": ["Limitation 1"]
}}

Be EXHAUSTIVE - extract every distinct component. For encoder-decoder models, ALWAYS include cross-attention as a separate component.'''


class PaperAnalyzer:
    """LLM-powered deep paper analysis"""

    def __init__(self, db: ArcFusionDB, api_key: str = None):
        self.db = db
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not HAS_ANTHROPIC:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required. Set environment variable or pass api_key parameter.")

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def analyze_paper(
        self,
        title: str,
        content: str,
        paper_id: str = "",
        model: str = "claude-sonnet-4-20250514",
        verbose: bool = True
    ) -> Optional[AnalysisResult]:
        """
        Deeply analyze a paper using Claude for exhaustive component extraction.

        Args:
            title: Paper title
            content: Abstract and/or full text
            paper_id: arXiv ID or other identifier
            model: Claude model to use
            verbose: Print progress

        Returns:
            AnalysisResult with extracted components
        """
        if verbose:
            print(f"Analyzing: {title[:60]}...")

        prompt = ANALYSIS_PROMPT.format(
            title=title,
            paper_id=paper_id,
            content=content[:15000]  # Limit content length
        )

        try:
            max_tokens = 8192
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )

            # Extract JSON from response with defensive checks
            if not response.content:
                if verbose:
                    print("  [ERROR] Empty response from LLM")
                return None

            response_text = response.content[0].text

            # Find JSON block (handle markdown code blocks)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                if json_end == -1:
                    if verbose:
                        print("  [ERROR] Malformed JSON code block (missing closing ```)")
                    return None
                response_text = response_text[json_start:json_end]
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                if json_end == -1:
                    if verbose:
                        print("  [ERROR] Malformed code block (missing closing ```)")
                    return None
                response_text = response_text[json_start:json_end]

            data = json.loads(response_text.strip())

            # Parse into structured result
            novel_components = []
            for i, comp_data in enumerate(data.get("novel_components", [])):
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
                    position=comp_data.get("position", i),  # Use index as fallback
                )
                novel_components.append(comp)

            # Sort by position to ensure correct architectural order
            novel_components.sort(key=lambda c: c.position)

            result = AnalysisResult(
                paper_title=title,
                paper_id=paper_id,
                architecture_name=data.get("architecture_name", title),
                architecture_description=data.get("architecture_description", ""),
                novel_components=novel_components,
                component_relationships=data.get("component_relationships", []),
                key_innovations=data.get("key_innovations", []),
                limitations=data.get("limitations", []),
            )

            if verbose:
                print(f"  Found {len(novel_components)} novel components")
                for comp in novel_components:
                    print(f"    [{comp.position}] {comp.name} (confidence: {comp.confidence:.0%})")

            return result

        except json.JSONDecodeError as e:
            if verbose:
                print(f"  [ERROR] Failed to parse JSON: {e}")
            return None
        except Exception as e:
            if verbose:
                print(f"  [ERROR] Analysis failed: {e}")
            return None

    def analyze_and_ingest(
        self,
        title: str,
        content: str,
        paper_id: str = "",
        paper_url: str = "",
        min_confidence: float = 0.7,
        verbose: bool = True
    ) -> tuple[Optional[Engine], list[Component]]:
        """
        Analyze paper and add novel components to database.

        Returns:
            (Engine, list of new Components created)
        """
        result = self.analyze_paper(title, content, paper_id, verbose=verbose)

        if not result:
            return None, []

        new_components = []
        component_ids = []

        # Check existing components to avoid duplicates
        existing = {c.name.lower(): c for c in self.db.find_components()}

        for analyzed in result.novel_components:
            # Skip low confidence
            if analyzed.confidence < min_confidence:
                if verbose:
                    print(f"  Skipping {analyzed.name} (confidence {analyzed.confidence:.0%} < {min_confidence:.0%})")
                continue

            # Check for duplicates (case-insensitive)
            if analyzed.name.lower() in existing:
                if verbose:
                    print(f"  Skipping {analyzed.name} (already exists)")
                component_ids.append(existing[analyzed.name.lower()].component_id)
                continue

            # Create new component
            comp = Component(
                name=analyzed.name,
                description=analyzed.description,
                interface_in=analyzed.interface_in,
                interface_out=analyzed.interface_out,
                code=analyzed.code_sketch,
                usefulness_score=analyzed.confidence,
                source_paper_id=paper_id,
                introduced_year=int(paper_id[:4]) if paper_id and paper_id[:4].isdigit() else 0,
                hyperparameters=analyzed.hyperparameters,
                time_complexity=analyzed.time_complexity,
                space_complexity=analyzed.space_complexity,
                flops_formula=analyzed.flops_formula,
                is_parallelizable=analyzed.is_parallelizable,
                is_causal=analyzed.is_causal,
                math_operations=analyzed.math_operations,
            )

            self.db.add_component(comp)
            new_components.append(comp)
            component_ids.append(comp.component_id)
            existing[comp.name.lower()] = comp

            if verbose:
                print(f"  [NEW] Added component: {comp.name}")

        # Create engine
        engine = Engine(
            name=result.architecture_name,
            description=result.architecture_description,
            paper_url=paper_url,
            engine_score=sum(c.confidence for c in result.novel_components) / max(len(result.novel_components), 1),
            component_ids=component_ids,
        )

        # Check if engine already exists
        if not self.db.get_engine_by_name(engine.name):
            self.db.add_engine(engine)
            if verbose:
                print(f"  [NEW] Added engine: {engine.name}")
        else:
            engine = self.db.get_engine_by_name(engine.name)
            if verbose:
                print(f"  Engine {engine.name} already exists")

        # Add component relationships (with validation)
        relationships_added = 0
        relationships_skipped = 0

        for rel in result.component_relationships:
            if len(rel) < 3:
                if verbose:
                    print(f"  [WARN] Invalid relationship format: {rel}")
                relationships_skipped += 1
                continue

            comp1_name, comp2_name, score = rel[0], rel[1], rel[2]
            comp1 = existing.get(comp1_name.lower())
            comp2 = existing.get(comp2_name.lower())

            # Validate both components exist before adding relationship
            if not comp1:
                if verbose:
                    print(f"  [WARN] Skipping relationship: '{comp1_name}' not found (may have been skipped due to low confidence)")
                relationships_skipped += 1
                continue

            if not comp2:
                if verbose:
                    print(f"  [WARN] Skipping relationship: '{comp2_name}' not found (may have been skipped due to low confidence)")
                relationships_skipped += 1
                continue

            # Validate score is in valid range
            if not isinstance(score, (int, float)) or score < 0 or score > 1:
                if verbose:
                    print(f"  [WARN] Invalid relationship score {score} for {comp1_name} <-> {comp2_name}, using 0.5")
                score = 0.5

            relationship = ComponentRelationship(
                component1_id=comp1.component_id,
                component2_id=comp2.component_id,
                engine_id=engine.engine_id,
                c2c_score=float(score),
            )
            self.db.add_relationship(relationship)
            relationships_added += 1

        if verbose and (relationships_added > 0 or relationships_skipped > 0):
            print(f"  Relationships: {relationships_added} added, {relationships_skipped} skipped")

        return engine, new_components

    def reanalyze_paper(self, arxiv_id: str, verbose: bool = True) -> tuple[Optional[Engine], list[Component]]:
        """
        Re-analyze a previously processed paper with deeper LLM analysis.
        Requires paper to be in processed_papers table with associated engine.
        """
        from .fetcher import ArxivFetcher

        fetcher = ArxivFetcher(self.db)
        paper = fetcher.fetch_by_id(arxiv_id)

        if not paper:
            if verbose:
                print(f"Could not fetch paper {arxiv_id}")
            return None, []

        return self.analyze_and_ingest(
            title=paper.title,
            content=paper.abstract,
            paper_id=paper.arxiv_id,
            paper_url=paper.pdf_url,
            verbose=verbose,
        )
