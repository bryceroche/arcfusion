"""
Paper Decomposer - Extract architecture components from papers.

Reads ML papers and decomposes their architectures into components.
"""

from .db import ArcFusionDB, Component, Engine, ComponentRelationship

# Default relationship score for components extracted from the same paper
# Lower than seed data (0.8) since extraction has more uncertainty
DEFAULT_EXTRACTED_RELATIONSHIP_SCORE = 0.7


class PaperDecomposer:
    """Extract architecture components from paper text."""

    COMPONENT_PATTERNS = {
        "attention": ["attention", "self-attention", "cross-attention", "multi-head"],
        "normalization": ["layer norm", "layernorm", "batch norm", "rmsnorm", "group norm"],
        "feedforward": ["feed-forward", "feedforward", "ffn", "mlp", "dense layer"],
        "embedding": ["embedding", "positional encoding", "token embedding"],
        "convolution": ["conv", "convolution", "cnn", "depthwise"],
        "pooling": ["pooling", "max pool", "avg pool", "global pool"],
        "activation": ["relu", "gelu", "swish", "silu", "mish", "softmax"],
        "dropout": ["dropout", "droppath", "stochastic depth"],
        "residual": ["residual", "skip connection", "shortcut"],
        "gating": ["gate", "glu", "swiglu", "geglu", "mixture of experts", "moe"],
        "ssm": ["state space", "ssm", "mamba", "s4", "selective state"],
        "retention": ["retention", "retnet"],
        "linear_attention": ["linear attention", "rwkv", "aft"],
    }

    # Map category names to search patterns for finding existing components
    CATEGORY_SEARCH_PATTERNS = {
        "attention": ["Attention", "attention", "MHA"],
        "normalization": ["Norm", "norm", "RMS"],
        "feedforward": ["Feed", "FFN", "MLP"],
        "embedding": ["Embed", "Position", "Token"],
        "convolution": ["Conv"],
        "pooling": ["Pool"],
        "activation": ["Activation", "GELU", "ReLU"],
        "dropout": ["Dropout"],
        "residual": ["Residual", "Skip"],
        "gating": ["Gate", "GLU", "MoE"],
        "ssm": ["SSM", "Mamba", "State"],
        "retention": ["Retention", "RetNet"],
        "linear_attention": ["Linear", "RWKV"],
    }

    def __init__(self, db: ArcFusionDB):
        self.db = db

    def extract_components_from_text(self, text: str) -> list[dict]:
        """Extract component mentions from text."""
        text_lower = text.lower()
        found_components = []

        for category, patterns in self.COMPONENT_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_lower:
                    count = text_lower.count(pattern)
                    confidence = min(1.0, count / 10)

                    found_components.append({
                        "category": category,
                        "pattern": pattern,
                        "confidence": confidence,
                        "occurrences": count
                    })

        best_per_category = {}
        for comp in found_components:
            cat = comp["category"]
            if cat not in best_per_category or comp["confidence"] > best_per_category[cat]["confidence"]:
                best_per_category[cat] = comp

        return list(best_per_category.values())

    def create_engine_from_paper(
        self,
        title: str,
        abstract: str,
        paper_url: str = "",
        full_text: str = ""
    ) -> tuple[Engine, list[Component]]:
        """Analyze a paper and create an Engine with its components."""
        all_text = f"{title}\n{abstract}\n{full_text}"
        extracted = self.extract_components_from_text(all_text)

        created_components = []
        comp_ids = []

        for ext in extracted:
            # Search for existing components using category-specific patterns
            existing = None
            search_patterns = self.CATEGORY_SEARCH_PATTERNS.get(ext["category"], [ext["category"]])
            for pattern in search_patterns:
                matches = self.db.find_components(pattern)
                if matches:
                    existing = matches[0]
                    break

            if existing:
                comp = existing
            else:
                comp = Component(
                    name=ext["category"].title(),
                    description=f"Component type: {ext['pattern']} (from {title})",
                    interface_in={"shape": "[batch, *, hidden]", "dtype": "float32"},
                    interface_out={"shape": "[batch, *, hidden]", "dtype": "float32"},
                    usefulness_score=ext["confidence"]
                )
                self.db.add_component(comp)
                created_components.append(comp)

            comp_ids.append(comp.component_id)

        engine = Engine(
            name=title[:100],
            description=abstract[:500],
            paper_url=paper_url,
            engine_score=0.5,
            component_ids=comp_ids
        )
        self.db.add_engine(engine)

        for i, cid1 in enumerate(comp_ids):
            for cid2 in comp_ids[i + 1:]:
                rel = ComponentRelationship(
                    component1_id=cid1,
                    component2_id=cid2,
                    engine_id=engine.engine_id,
                    c2c_score=DEFAULT_EXTRACTED_RELATIONSHIP_SCORE
                )
                self.db.add_relationship(rel)

        return engine, created_components
