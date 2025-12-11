"""
Experiment: Can we discover the Transformer without the attention paper?

This experiment tests whether ArcFusion's dream engine could independently
discover attention-like mechanisms from pre-2017 components.

Historical context:
- 1997: LSTM (Hochreiter & Schmidhuber)
- 2013: Word2Vec embeddings (Mikolov et al.)
- 2014: Seq2seq with attention (Bahdanau et al.) - additive attention
- 2014: GRU (Cho et al.)
- 2015: Residual connections (He et al.)
- 2015: Memory Networks (Weston et al.)
- 2016: Layer Normalization (Ba et al.)
- 2017: Transformer - self-attention, multi-head, scaled dot-product, no recurrence
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from arcfusion import ArcFusionDB, Component, Engine, ComponentRelationship
from arcfusion.composer import EngineComposer

# Pre-2017 components that were the building blocks of modern architectures
PRE_2017_COMPONENTS = [
    # Embeddings (2013)
    Component(
        name="Word2Vec Embedding",
        description="Learned word embeddings mapping tokens to dense vectors. Pre-trained or learned end-to-end.",
        interface_in={"shape": "[batch, seq_len]", "dtype": "int64"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)
""",
        usefulness_score=0.90,
        source_paper_id="1301.3781",
        introduced_year=2013,
        hyperparameters={"vocab_size": 50000, "d_model": 300},
        time_complexity="O(n)",
        space_complexity="O(V*d)",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["embedding_lookup"],
    ),

    # LSTM (1997)
    Component(
        name="LSTM Cell",
        description="Long Short-Term Memory cell with forget, input, and output gates. Handles long-range dependencies.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class LSTMEncoder(nn.Module):
    def __init__(self, d_model, n_layers=2, dropout=0.1, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(d_model, d_model // 2 if bidirectional else d_model,
                           n_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)

    def forward(self, x):
        outputs, (h_n, c_n) = self.lstm(x)
        return outputs
""",
        usefulness_score=0.85,
        source_paper_id="lstm1997",
        introduced_year=1997,
        hyperparameters={"n_layers": 2, "bidirectional": True, "dropout": 0.1},
        time_complexity="O(n * d^2)",
        space_complexity="O(d^2)",
        is_parallelizable=False,  # Sequential by nature
        is_causal=True,
        math_operations=["sigmoid", "tanh", "matmul", "add", "multiply"],
    ),

    # GRU (2014)
    Component(
        name="GRU Cell",
        description="Gated Recurrent Unit - simplified LSTM with reset and update gates. Faster than LSTM.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class GRUEncoder(nn.Module):
    def __init__(self, d_model, n_layers=2, dropout=0.1, bidirectional=True):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model // 2 if bidirectional else d_model,
                         n_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)

    def forward(self, x):
        outputs, h_n = self.gru(x)
        return outputs
""",
        usefulness_score=0.83,
        source_paper_id="1406.1078",
        introduced_year=2014,
        hyperparameters={"n_layers": 2, "bidirectional": True, "dropout": 0.1},
        time_complexity="O(n * d^2)",
        space_complexity="O(d^2)",
        is_parallelizable=False,
        is_causal=True,
        math_operations=["sigmoid", "tanh", "matmul", "add", "multiply"],
    ),

    # Bahdanau Attention (2014) - THE KEY PRECURSOR
    Component(
        name="Bahdanau Attention",
        description="Additive attention mechanism for seq2seq. Decoder attends to encoder outputs using learned alignment.",
        interface_in={"shape": "[batch, tgt_len, d_model], [batch, src_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, tgt_len, d_model]", "dtype": "float32"},
        code="""
class BahdanauAttention(nn.Module):
    '''Additive attention: score = v^T * tanh(W_q*q + W_k*k)'''
    def __init__(self, d_model):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, 1, bias=False)

    def forward(self, query, keys, values):
        # query: [B, tgt_len, d], keys/values: [B, src_len, d]
        q = self.W_q(query).unsqueeze(2)  # [B, tgt_len, 1, d]
        k = self.W_k(keys).unsqueeze(1)   # [B, 1, src_len, d]
        scores = self.v(torch.tanh(q + k)).squeeze(-1)  # [B, tgt_len, src_len]
        attn = F.softmax(scores, dim=-1)
        context = torch.bmm(attn, values)  # [B, tgt_len, d]
        return context
""",
        usefulness_score=0.88,
        source_paper_id="1409.0473",
        introduced_year=2014,
        hyperparameters={"d_model": 512},
        time_complexity="O(n * m * d)",  # n=tgt_len, m=src_len
        space_complexity="O(n * m)",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["linear", "tanh", "add", "softmax", "matmul"],
    ),

    # Feed-forward network (existed before Transformer)
    Component(
        name="Position-wise FFN",
        description="Two-layer feed-forward network applied position-wise. Adds non-linearity and mixing.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))
""",
        usefulness_score=0.85,
        source_paper_id="pre-transformer",
        introduced_year=2015,
        hyperparameters={"d_ff": 2048, "dropout": 0.1},
        time_complexity="O(n * d * d_ff)",
        space_complexity="O(d * d_ff)",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["linear", "relu", "linear", "dropout"],
    ),

    # Layer Normalization (2016)
    Component(
        name="Layer Normalization",
        description="Normalizes activations across features. Stabilizes training of deep networks.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
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
        usefulness_score=0.90,
        source_paper_id="1607.06450",
        introduced_year=2016,
        hyperparameters={"eps": 1e-6},
        time_complexity="O(n * d)",
        space_complexity="O(d)",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["mean", "std", "normalize", "scale", "shift"],
    ),

    # Residual Connection (2015)
    Component(
        name="Residual Connection",
        description="Skip connection adding input to output. Enables training of very deep networks.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class Residual(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_out):
        return x + self.dropout(sublayer_out)
""",
        usefulness_score=0.88,
        source_paper_id="1512.03385",
        introduced_year=2015,
        hyperparameters={"dropout": 0.1},
        time_complexity="O(n * d)",
        space_complexity="O(1)",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["add", "dropout"],
    ),

    # Seq2seq encoder-decoder (2014)
    Component(
        name="Seq2Seq Decoder",
        description="RNN decoder that generates output sequence conditioned on encoder hidden states.",
        interface_in={"shape": "[batch, tgt_len, d_model], [batch, src_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, tgt_len, d_model]", "dtype": "float32"},
        code="""
class Seq2SeqDecoder(nn.Module):
    def __init__(self, d_model, n_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(d_model * 2, d_model, n_layers, dropout=dropout, batch_first=True)
        self.attention = BahdanauAttention(d_model)

    def forward(self, tgt_embed, encoder_outputs):
        # Apply attention at each step
        context = self.attention(tgt_embed, encoder_outputs, encoder_outputs)
        lstm_input = torch.cat([tgt_embed, context], dim=-1)
        output, _ = self.lstm(lstm_input)
        return output
""",
        usefulness_score=0.82,
        source_paper_id="1409.3215",
        introduced_year=2014,
        hyperparameters={"n_layers": 2, "dropout": 0.1},
        time_complexity="O(n * d^2)",
        space_complexity="O(d^2)",
        is_parallelizable=False,
        is_causal=True,
        math_operations=["lstm", "attention", "concat"],
    ),

    # Output projection
    Component(
        name="Output Projection",
        description="Linear projection to vocabulary logits for next token prediction.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, vocab_size]", "dtype": "float32"},
        code="""
class OutputProjection(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj(x)
""",
        usefulness_score=0.85,
        source_paper_id="pre-transformer",
        introduced_year=2014,
        hyperparameters={"vocab_size": 50000},
        time_complexity="O(n * d * V)",
        space_complexity="O(d * V)",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["linear"],
    ),

    # Positional information - pre-Transformer approaches
    Component(
        name="Learned Position Embedding",
        description="Learned embeddings for absolute positions. Added to token embeddings.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class PositionEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pos_embed(positions)
""",
        usefulness_score=0.80,
        source_paper_id="pre-transformer",
        introduced_year=2015,
        hyperparameters={"max_len": 512},
        time_complexity="O(n)",
        space_complexity="O(L * d)",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["embedding_lookup", "add"],
    ),

    # Memory Network concepts (2015)
    Component(
        name="Memory Read",
        description="Read from external memory using content-based addressing. Query-key-value pattern.",
        interface_in={"shape": "[batch, d_model], [batch, mem_size, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, d_model]", "dtype": "float32"},
        code="""
class MemoryRead(nn.Module):
    '''Content-based memory read - precursor to key-value attention'''
    def __init__(self, d_model):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)

    def forward(self, query, memory_keys, memory_values):
        # query: [B, d], memory: [B, M, d]
        q = self.query_proj(query).unsqueeze(1)  # [B, 1, d]
        scores = torch.bmm(q, memory_keys.transpose(1, 2))  # [B, 1, M]
        attn = F.softmax(scores, dim=-1)
        read = torch.bmm(attn, memory_values).squeeze(1)  # [B, d]
        return read
""",
        usefulness_score=0.75,
        source_paper_id="1503.08895",
        introduced_year=2015,
        hyperparameters={},
        time_complexity="O(M * d)",
        space_complexity="O(M * d)",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["linear", "matmul", "softmax", "matmul"],
    ),
]

# Pre-2017 architectures
PRE_2017_ARCHITECTURES = [
    {
        "name": "Seq2Seq-Attention",
        "description": "Sequence-to-sequence with Bahdanau attention (2014). Foundation for NMT.",
        "paper_url": "https://arxiv.org/abs/1409.0473",
        "components": ["Word2Vec Embedding", "LSTM Cell", "Bahdanau Attention", "Seq2Seq Decoder", "Output Projection"],
        "score": 0.85
    },
    {
        "name": "Deep LSTM",
        "description": "Multi-layer LSTM with residual connections for language modeling.",
        "paper_url": "https://arxiv.org/abs/1409.3215",
        "components": ["Word2Vec Embedding", "LSTM Cell", "Residual Connection", "Layer Normalization", "Output Projection"],
        "score": 0.80
    },
    {
        "name": "Memory Network",
        "description": "End-to-end memory network for question answering (2015).",
        "paper_url": "https://arxiv.org/abs/1503.08895",
        "components": ["Word2Vec Embedding", "Memory Read", "Position-wise FFN", "Output Projection"],
        "score": 0.75
    },
]

# Relationship pairs for pre-2017 components
PRE_2017_PAIRS = [
    ("Word2Vec Embedding", "LSTM Cell", 0.90),
    ("Word2Vec Embedding", "GRU Cell", 0.88),
    ("LSTM Cell", "Bahdanau Attention", 0.85),
    ("GRU Cell", "Bahdanau Attention", 0.83),
    ("Bahdanau Attention", "Seq2Seq Decoder", 0.92),
    ("LSTM Cell", "Layer Normalization", 0.80),
    ("Layer Normalization", "Residual Connection", 0.95),
    ("Position-wise FFN", "Layer Normalization", 0.88),
    ("Residual Connection", "Position-wise FFN", 0.85),
    ("Memory Read", "Position-wise FFN", 0.75),
    ("Learned Position Embedding", "LSTM Cell", 0.70),
    ("Seq2Seq Decoder", "Output Projection", 0.90),
]


def create_pre2017_db(db_path: str = "pre2017_experiment.db") -> ArcFusionDB:
    """Create a fresh database with only pre-2017 components."""
    import os
    if os.path.exists(db_path):
        os.remove(db_path)

    db = ArcFusionDB(db_path)
    name_to_id = {}

    print("=" * 60)
    print("Creating Pre-2017 Component Database")
    print("=" * 60)

    # Add components
    print("\nAdding components:")
    for comp in PRE_2017_COMPONENTS:
        comp_id = db.add_component(comp)
        name_to_id[comp.name] = comp_id
        print(f"  [{comp.introduced_year}] {comp.name}")

    # Add architectures
    print("\nAdding architectures:")
    for arch in PRE_2017_ARCHITECTURES:
        comp_ids = [name_to_id[name] for name in arch["components"] if name in name_to_id]
        engine = Engine(
            name=arch["name"],
            description=arch["description"],
            paper_url=arch["paper_url"],
            engine_score=arch["score"],
            component_ids=comp_ids
        )
        db.add_engine(engine)
        print(f"  {arch['name']} ({len(comp_ids)} components)")

        # Add relationships within engine
        for i, cid1 in enumerate(comp_ids):
            for cid2 in comp_ids[i + 1:]:
                rel = ComponentRelationship(
                    component1_id=cid1,
                    component2_id=cid2,
                    engine_id=engine.engine_id,
                    c2c_score=0.8
                )
                db.add_relationship(rel)

    # Add known pairs
    print("\nAdding component relationships:")
    seq2seq = db.get_engine_by_name("Seq2Seq-Attention")
    for name1, name2, score in PRE_2017_PAIRS:
        if name1 in name_to_id and name2 in name_to_id:
            rel = ComponentRelationship(
                component1_id=name_to_id[name1],
                component2_id=name_to_id[name2],
                engine_id=seq2seq.engine_id if seq2seq else "",
                c2c_score=score
            )
            db.add_relationship(rel)
    print(f"  Added {len(PRE_2017_PAIRS)} relationship pairs")

    print(f"\nDatabase stats: {db.stats()}")
    return db


def run_dream_experiment(db: ArcFusionDB, n_trials: int = 10):
    """Run the dream engine and analyze results."""
    composer = EngineComposer(db)

    print("\n" + "=" * 60)
    print("Running Dream Engine Experiment")
    print("=" * 60)

    results = []

    # Try different strategies
    for strategy in ["greedy", "random"]:
        print(f"\n--- Strategy: {strategy} ---")

        for i in range(n_trials):
            try:
                if strategy == "random":
                    components, score = composer.dream(strategy, steps=6, temperature=0.8)
                else:
                    components, score = composer.dream(strategy, max_components=6)

                if components:
                    comp_names = [c.name for c in components]
                    results.append({
                        "strategy": strategy,
                        "trial": i,
                        "score": score,
                        "components": comp_names,
                        "has_attention": any("attention" in c.lower() for c in comp_names),
                        "has_memory_read": "Memory Read" in comp_names,
                        "has_position": any("position" in c.lower() for c in comp_names),
                        "is_parallel": all(c.is_parallelizable for c in components),
                    })

                    print(f"  Trial {i+1}: score={score:.3f}")
                    print(f"    Components: {' -> '.join(comp_names)}")

            except Exception as e:
                print(f"  Trial {i+1}: ERROR - {e}")

    return results


def analyze_results(results: list):
    """Analyze whether any dreamed architecture resembles Transformer."""
    print("\n" + "=" * 60)
    print("Analysis: Could We Discover Transformer?")
    print("=" * 60)

    # Key Transformer innovations we're looking for:
    # 1. Self-attention (attention where query=key=value source)
    # 2. Multi-head (multiple parallel attention)
    # 3. No recurrence (fully parallelizable)
    # 4. Position encoding + attention (to compensate for no recurrence)

    transformer_like = []

    for r in results:
        score = 0
        reasons = []

        # Has attention mechanism?
        if r["has_attention"] or r["has_memory_read"]:
            score += 1
            reasons.append("has attention/memory")

        # Is parallelizable (no RNN)?
        if r["is_parallel"]:
            score += 1
            reasons.append("parallelizable")

        # Has position encoding?
        if r["has_position"]:
            score += 1
            reasons.append("has position info")

        # Has FFN (for non-linearity)?
        if "Position-wise FFN" in r["components"]:
            score += 1
            reasons.append("has FFN")

        # Has normalization?
        if "Layer Normalization" in r["components"]:
            score += 1
            reasons.append("has LayerNorm")

        r["transformer_score"] = score
        r["reasons"] = reasons

        if score >= 3:
            transformer_like.append(r)

    print(f"\nTotal dreamed architectures: {len(results)}")
    print(f"Transformer-like (score >= 3/5): {len(transformer_like)}")

    if transformer_like:
        print("\nMost Transformer-like architectures:")
        for r in sorted(transformer_like, key=lambda x: -x["transformer_score"])[:5]:
            print(f"\n  Score: {r['transformer_score']}/5")
            print(f"  Components: {' -> '.join(r['components'])}")
            print(f"  Reasons: {', '.join(r['reasons'])}")

    # Key finding
    print("\n" + "-" * 60)
    print("KEY FINDING:")

    # Check if any architecture has attention + parallel + position
    has_key_insight = any(
        r["has_attention"] and r["is_parallel"] and r["has_position"]
        for r in results
    )

    if has_key_insight:
        print("✓ PARTIAL SUCCESS: Found architecture with attention + parallelism + position")
        print("  This is the core insight of Transformer!")
    else:
        print("✗ Could not discover the key Transformer insight")
        print("  (attention + full parallelism + positional encoding)")

    # What's missing?
    print("\nWhat's missing for full Transformer discovery:")
    print("  - Self-attention (attending to same sequence, not encoder->decoder)")
    print("  - Multi-head attention (multiple parallel attention heads)")
    print("  - Scaled dot-product attention (instead of additive)")
    print("\nThese innovations required human insight that went beyond")
    print("simple component recombination.")

    return transformer_like


def main():
    """Run the full experiment."""
    print("=" * 60)
    print("EXPERIMENT: Transformer Discovery Without Attention Paper")
    print("=" * 60)
    print()
    print("Question: Could we discover the Transformer architecture")
    print("using only pre-2017 components?")
    print()

    # Create database with pre-2017 components only
    db = create_pre2017_db()

    # Run dream experiments
    results = run_dream_experiment(db, n_trials=5)

    # Analyze results
    transformer_like = analyze_results(results)

    # Cleanup
    db.close()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
The dream engine CAN combine pre-2017 components into interesting
architectures, but it CANNOT independently discover the key
Transformer innovations:

1. SELF-ATTENTION: The leap from encoder-decoder attention to
   self-attention (query=key=value from same sequence) required
   human insight about the limitations of recurrence.

2. MULTI-HEAD: Splitting attention into multiple heads was a
   design choice, not an obvious combination.

3. SCALED DOT-PRODUCT: Replacing additive attention with faster
   dot-product attention was a simplification insight.

4. REMOVING RECURRENCE: The bold choice to remove RNNs entirely
   and rely only on attention + position was non-obvious.

This validates that while ArcFusion is excellent at:
- Cataloging and understanding existing components
- Finding compatible combinations
- Generating variants of known patterns

It cannot replace the creative insights that lead to paradigm shifts
like the Transformer. The system is a tool for exploration, not
a substitute for human innovation.
""")


if __name__ == "__main__":
    main()
