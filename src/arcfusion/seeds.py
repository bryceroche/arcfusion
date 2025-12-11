"""
Seed Data - Pre-defined components and architectures.

Includes Transformer components and modern architectures (Mamba, RWKV, RetNet, etc.)
"""

from .db import ArcFusionDB, Component, Engine, ComponentRelationship


TRANSFORMER_COMPONENTS = [
    Component(
        name="Embedding",
        description="Token + positional embeddings. Maps discrete tokens to continuous vectors.",
        interface_in={"shape": "[batch, seq_len]", "dtype": "int64"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return self.token_embed(x) + self.pos_embed(positions)
""",
        usefulness_score=0.95
    ),
    Component(
        name="MultiHeadAttention",
        description="Scaled dot-product attention with multiple heads. Core innovation of Transformer.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)
""",
        usefulness_score=0.98
    ),
    Component(
        name="FeedForward",
        description="Position-wise feed-forward network. Two linear transforms with activation.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))
""",
        usefulness_score=0.90
    ),
    Component(
        name="LayerNorm",
        description="Layer normalization. Stabilizes training by normalizing activations.",
        interface_in={"shape": "[batch, *, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, *, d_model]", "dtype": "float32"},
        code="""
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
""",
        usefulness_score=0.92
    ),
    Component(
        name="ResidualConnection",
        description="Skip connection with dropout. Enables deep networks by gradient flow.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        usefulness_score=0.88
    ),
    Component(
        name="SoftmaxOutput",
        description="Final projection to vocabulary logits with softmax.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, vocab_size]", "dtype": "float32"},
        usefulness_score=0.85
    ),
    Component(
        name="CausalMask",
        description="Autoregressive mask preventing attention to future tokens.",
        interface_in={"shape": "[seq_len]", "dtype": "int64"},
        interface_out={"shape": "[seq_len, seq_len]", "dtype": "bool"},
        usefulness_score=0.80
    ),
]

MODERN_COMPONENTS = [
    Component(
        name="SelectiveSSM",
        description="Selective State Space Model. Input-dependent state transitions (Mamba).",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        usefulness_score=0.88
    ),
    Component(
        name="RetentionHead",
        description="Retention mechanism with parallel/recurrent/chunk modes (RetNet).",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        usefulness_score=0.82
    ),
    Component(
        name="TimeMixing",
        description="RWKV time-mixing: linear attention with time-based token mixing.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        usefulness_score=0.80
    ),
    Component(
        name="ChannelMixing",
        description="RWKV channel-mixing: position-wise feed-forward with gating.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        usefulness_score=0.78
    ),
    Component(
        name="RMSNorm",
        description="Root Mean Square normalization. Simpler than LayerNorm, used in LLaMA.",
        interface_in={"shape": "[batch, *, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, *, d_model]", "dtype": "float32"},
        usefulness_score=0.91
    ),
    Component(
        name="RotaryEmbedding",
        description="Rotary Position Embedding (RoPE). Encodes position in rotation.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        usefulness_score=0.89
    ),
    Component(
        name="SwiGLU",
        description="Swish-Gated Linear Unit. Better FFN activation from PaLM/LLaMA.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        usefulness_score=0.87
    ),
    Component(
        name="GroupedQueryAttention",
        description="GQA: Multiple query heads share key-value heads. Memory efficient.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        usefulness_score=0.86
    ),
]

ARCHITECTURES = [
    {
        "name": "Transformer",
        "description": "Attention Is All You Need (Vaswani et al., 2017). Foundation for modern LLMs.",
        "paper_url": "https://arxiv.org/abs/1706.03762",
        "components": ["Embedding", "MultiHeadAttention", "FeedForward", "LayerNorm", "ResidualConnection", "CausalMask", "SoftmaxOutput"],
        "score": 0.99
    },
    {
        "name": "GPT-2",
        "description": "Decoder-only transformer with causal masking. Trained on WebText.",
        "paper_url": "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
        "components": ["Embedding", "MultiHeadAttention", "FeedForward", "LayerNorm", "CausalMask", "SoftmaxOutput"],
        "score": 0.95
    },
    {
        "name": "BERT",
        "description": "Bidirectional encoder with masked language modeling.",
        "paper_url": "https://arxiv.org/abs/1810.04805",
        "components": ["Embedding", "MultiHeadAttention", "FeedForward", "LayerNorm", "SoftmaxOutput"],
        "score": 0.94
    },
    {
        "name": "LLaMA",
        "description": "Meta's efficient LLM with RMSNorm, RoPE, and SwiGLU.",
        "paper_url": "https://arxiv.org/abs/2302.13971",
        "components": ["Embedding", "MultiHeadAttention", "RMSNorm", "RotaryEmbedding", "SwiGLU", "CausalMask", "SoftmaxOutput"],
        "score": 0.96
    },
    {
        "name": "Mamba",
        "description": "State space model replacing attention with selective SSM.",
        "paper_url": "https://arxiv.org/abs/2312.00752",
        "components": ["Embedding", "SelectiveSSM", "LayerNorm", "SoftmaxOutput"],
        "score": 0.90
    },
    {
        "name": "RetNet",
        "description": "Retention mechanism as attention alternative with linear complexity.",
        "paper_url": "https://arxiv.org/abs/2307.08621",
        "components": ["Embedding", "RetentionHead", "FeedForward", "LayerNorm", "SoftmaxOutput"],
        "score": 0.85
    },
    {
        "name": "RWKV",
        "description": "Linear attention RNN achieving transformer-level performance.",
        "paper_url": "https://arxiv.org/abs/2305.13048",
        "components": ["Embedding", "TimeMixing", "ChannelMixing", "LayerNorm", "SoftmaxOutput"],
        "score": 0.86
    },
    {
        "name": "Mistral-7B",
        "description": "Efficient 7B model with GQA and sliding window attention.",
        "paper_url": "https://arxiv.org/abs/2310.06825",
        "components": ["Embedding", "GroupedQueryAttention", "RMSNorm", "RotaryEmbedding", "SwiGLU", "CausalMask", "SoftmaxOutput"],
        "score": 0.93
    },
]

COMPONENT_PAIRS = [
    ("Embedding", "MultiHeadAttention", 0.95),
    ("MultiHeadAttention", "FeedForward", 0.97),
    ("MultiHeadAttention", "LayerNorm", 0.96),
    ("FeedForward", "LayerNorm", 0.94),
    ("LayerNorm", "ResidualConnection", 0.98),
    ("ResidualConnection", "MultiHeadAttention", 0.93),
    ("FeedForward", "SoftmaxOutput", 0.88),
    ("CausalMask", "MultiHeadAttention", 0.92),
    ("RMSNorm", "MultiHeadAttention", 0.95),
    ("RotaryEmbedding", "MultiHeadAttention", 0.94),
    ("SwiGLU", "RMSNorm", 0.91),
    ("GroupedQueryAttention", "RMSNorm", 0.93),
    ("SelectiveSSM", "LayerNorm", 0.89),
    ("TimeMixing", "ChannelMixing", 0.92),
    ("RetentionHead", "FeedForward", 0.87),
]


def seed_transformers(db: ArcFusionDB, verbose: bool = True) -> dict[str, str]:
    """Seed database with Transformer components. Returns name->id mapping."""
    name_to_id = {}

    for comp in TRANSFORMER_COMPONENTS:
        comp_id = db.add_component(comp)
        name_to_id[comp.name] = comp_id
        if verbose:
            print(f"  Added component: {comp.name}")

    return name_to_id


def seed_modern_architectures(db: ArcFusionDB, verbose: bool = True) -> None:
    """Seed database with modern architectures (Mamba, RWKV, LLaMA, etc.)"""

    # First get existing components
    name_to_id = {}
    for comp in db.find_components():
        name_to_id[comp.name] = comp.component_id

    # Add modern components
    for comp in MODERN_COMPONENTS:
        if comp.name not in name_to_id:
            comp_id = db.add_component(comp)
            name_to_id[comp.name] = comp_id
            if verbose:
                print(f"  Added component: {comp.name}")

    # Add architectures
    for arch in ARCHITECTURES:
        existing = db.get_engine_by_name(arch["name"])
        if existing:
            if verbose:
                print(f"  Skipping {arch['name']} (exists)")
            continue

        comp_ids = [name_to_id[n] for n in arch["components"] if n in name_to_id]

        engine = Engine(
            name=arch["name"],
            description=arch["description"],
            paper_url=arch["paper_url"],
            engine_score=arch["score"],
            component_ids=comp_ids
        )
        db.add_engine(engine)
        if verbose:
            print(f"  Added engine: {arch['name']} ({len(comp_ids)} components)")

        # Record relationships
        for i, cid1 in enumerate(comp_ids):
            for cid2 in comp_ids[i + 1:]:
                rel = ComponentRelationship(
                    component1_id=cid1,
                    component2_id=cid2,
                    engine_id=engine.engine_id,
                    c2c_score=0.8
                )
                db.add_relationship(rel)

    # Add known good pairs
    for name1, name2, score in COMPONENT_PAIRS:
        if name1 in name_to_id and name2 in name_to_id:
            # Use Transformer as reference engine for pairs
            transformer = db.get_engine_by_name("Transformer")
            if transformer:
                rel = ComponentRelationship(
                    component1_id=name_to_id[name1],
                    component2_id=name_to_id[name2],
                    engine_id=transformer.engine_id,
                    c2c_score=score
                )
                db.add_relationship(rel)
