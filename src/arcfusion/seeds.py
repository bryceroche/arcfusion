"""
Seed Data - Pre-defined components and architectures.

Includes Transformer components and modern architectures (Mamba, RWKV, RetNet, etc.)
"""

from .db import ArcFusionDB, Component, Engine, ComponentRelationship

# Relationship score for components in well-known architectures
# Higher than extracted (0.7) since these are verified working combinations
DEFAULT_SEED_RELATIONSHIP_SCORE = 0.8


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
        usefulness_score=0.95,
        source_paper_id="1706.03762",
        introduced_year=2017,
        hyperparameters={"vocab_size": 50257, "d_model": 512, "max_seq_len": 512},
        time_complexity="O(n)",
        space_complexity="O(V*d + L*d)",
        flops_formula="n * d (lookup)",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["embedding_lookup", "add"],
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
        usefulness_score=0.98,
        source_paper_id="1706.03762",
        introduced_year=2017,
        hyperparameters={"n_heads": 8, "d_model": 512, "d_k": 64},
        time_complexity="O(n^2 * d)",
        space_complexity="O(n^2 + n*d)",
        flops_formula="4*n*d^2 + 2*n^2*d",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["matmul", "scale", "softmax", "matmul", "linear"],
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
        usefulness_score=0.90,
        source_paper_id="1706.03762",
        introduced_year=2017,
        hyperparameters={"d_model": 512, "d_ff": 2048, "activation": "gelu"},
        time_complexity="O(n * d * d_ff)",
        space_complexity="O(d * d_ff)",
        flops_formula="2 * n * d * d_ff",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["linear", "gelu", "linear"],
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
        usefulness_score=0.92,
        source_paper_id="1607.06450",
        introduced_year=2016,
        hyperparameters={"eps": 1e-6},
        time_complexity="O(n * d)",
        space_complexity="O(d)",
        flops_formula="5 * n * d",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["mean", "std", "subtract", "divide", "scale", "add"],
    ),
    Component(
        name="ResidualConnection",
        description="Skip connection with dropout. Enables deep networks by gradient flow.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class ResidualConnection(nn.Module):
    '''Pre-norm residual connection with dropout'''
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        '''x + Dropout(sublayer(x))'''
        return x + self.dropout(sublayer_output)
""",
        usefulness_score=0.88,
        source_paper_id="1512.03385",
        introduced_year=2015,
        hyperparameters={"dropout": 0.1},
        time_complexity="O(n * d)",
        space_complexity="O(1)",
        flops_formula="n * d",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["add", "dropout"],
    ),
    Component(
        name="SoftmaxOutput",
        description="Final projection to vocabulary logits with softmax.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, vocab_size]", "dtype": "float32"},
        code="""
class SoftmaxOutput(nn.Module):
    '''Language model head: project to vocab and apply softmax'''
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, temperature=1.0):
        logits = self.proj(x)
        return F.softmax(logits / temperature, dim=-1)

    def get_logits(self, x):
        '''Return raw logits (for cross-entropy loss)'''
        return self.proj(x)
""",
        usefulness_score=0.85,
        source_paper_id="1706.03762",
        introduced_year=2017,
        hyperparameters={"vocab_size": 50257},
        time_complexity="O(n * d * V)",
        space_complexity="O(d * V)",
        flops_formula="n * d * V",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["linear", "softmax"],
    ),
    Component(
        name="CausalMask",
        description="Autoregressive mask preventing attention to future tokens.",
        interface_in={"shape": "[seq_len]", "dtype": "int64"},
        interface_out={"shape": "[seq_len, seq_len]", "dtype": "bool"},
        code="""
class CausalMask(nn.Module):
    '''Generate causal (autoregressive) attention mask'''
    def __init__(self, max_seq_len=2048):
        super().__init__()
        # Pre-compute mask for efficiency
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, seq_len):
        '''Returns mask where True = masked (don't attend)'''
        return self.mask[:seq_len, :seq_len]

    @staticmethod
    def apply_mask(scores, mask):
        '''Apply causal mask to attention scores'''
        return scores.masked_fill(mask, float('-inf'))
""",
        usefulness_score=0.80,
        source_paper_id="1706.03762",
        introduced_year=2017,
        time_complexity="O(n^2)",
        space_complexity="O(n^2)",
        flops_formula="n^2",
        is_parallelizable=True,
        is_causal=True,
        math_operations=["triu", "mask_fill"],
    ),
]

MODERN_COMPONENTS = [
    Component(
        name="SelectiveSSM",
        description="Selective State Space Model. Input-dependent state transitions (Mamba).",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class SelectiveSSM(nn.Module):
    '''Simplified Mamba-style selective state space model'''
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv-1, groups=self.d_inner)

        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # dt, B, C
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Learnable A (log-space for stability)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        x = F.silu(x)

        # Selective scan (simplified)
        y = x * F.silu(z)  # Gating
        return self.out_proj(y)
""",
        usefulness_score=0.88,
        source_paper_id="2312.00752",
        introduced_year=2023,
        hyperparameters={"d_state": 16, "d_conv": 4, "expand": 2},
        time_complexity="O(n)",
        space_complexity="O(n * d_state)",
        flops_formula="n * d * d_state",
        is_parallelizable=False,  # Sequential by nature
        is_causal=True,
        math_operations=["conv1d", "ssm_scan", "silu", "linear"],
    ),
    Component(
        name="RetentionHead",
        description="Retention mechanism with parallel/recurrent/chunk modes (RetNet).",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class RetentionHead(nn.Module):
    '''Single retention head (parallel mode for training)'''
    def __init__(self, d_model, n_heads=8, gamma=0.99):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.gamma = gamma

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        q = self.W_q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Decay matrix D[i,j] = gamma^(i-j) for i >= j, else 0
        decay = torch.tril(self.gamma ** torch.arange(L).view(-1, 1) / self.gamma ** torch.arange(L).view(1, -1))
        decay = decay.to(x.device)

        # Retention: (Q @ K^T) * D @ V
        retention = (q @ k.transpose(-2, -1)) * decay
        out = retention @ v
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)
""",
        usefulness_score=0.82,
        source_paper_id="2307.08621",
        introduced_year=2023,
        hyperparameters={"n_heads": 8, "gamma": 0.99},
        time_complexity="O(n)",  # Recurrent mode
        space_complexity="O(n * d)",
        flops_formula="n * d^2 / h",
        is_parallelizable=True,  # Parallel mode available
        is_causal=True,
        math_operations=["linear", "retention_decay", "matmul"],
    ),
    Component(
        name="TimeMixing",
        description="RWKV time-mixing: linear attention with time-based token mixing.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class TimeMixing(nn.Module):
    '''RWKV time-mixing (simplified WKV computation)'''
    def __init__(self, d_model):
        super().__init__()
        self.time_decay = nn.Parameter(torch.zeros(d_model))  # w in paper
        self.time_first = nn.Parameter(torch.zeros(d_model))  # u in paper
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)

        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_r = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        # Shift for time mixing
        x_prev = F.pad(x, (0, 0, 1, -1))

        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_prev * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)

        k = self.W_k(xk)
        v = self.W_v(xv)
        r = torch.sigmoid(self.W_r(xr))

        # Simplified WKV (full version uses recurrent scan)
        wkv = k * v  # Simplified: should be cumulative with decay
        return self.W_o(r * wkv)
""",
        usefulness_score=0.80,
        source_paper_id="2305.13048",
        introduced_year=2023,
        hyperparameters={"time_decay": "learned", "time_first": "learned"},
        time_complexity="O(n)",
        space_complexity="O(d)",
        flops_formula="n * d",
        is_parallelizable=False,
        is_causal=True,
        math_operations=["linear", "sigmoid", "exp", "cumsum"],
    ),
    Component(
        name="ChannelMixing",
        description="RWKV channel-mixing: position-wise feed-forward with gating.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class ChannelMixing(nn.Module):
    '''RWKV channel-mixing (gated FFN)'''
    def __init__(self, d_model, d_ff_multiplier=4):
        super().__init__()
        d_ff = int(d_model * d_ff_multiplier)
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)

        self.W_k = nn.Linear(d_model, d_ff, bias=False)
        self.W_v = nn.Linear(d_ff, d_model, bias=False)
        self.W_r = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        x_prev = F.pad(x, (0, 0, 1, -1))

        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)

        k = torch.square(torch.relu(self.W_k(xk)))  # Squared ReLU
        r = torch.sigmoid(self.W_r(xr))
        return r * self.W_v(k)
""",
        usefulness_score=0.78,
        source_paper_id="2305.13048",
        introduced_year=2023,
        hyperparameters={"d_ff_multiplier": 4},
        time_complexity="O(n * d * d_ff)",
        space_complexity="O(d * d_ff)",
        flops_formula="2 * n * d * d_ff",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["linear", "sigmoid", "multiply", "linear"],
    ),
    Component(
        name="RMSNorm",
        description="Root Mean Square normalization. Simpler than LayerNorm, used in LLaMA.",
        interface_in={"shape": "[batch, *, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, *, d_model]", "dtype": "float32"},
        code="""
class RMSNorm(nn.Module):
    '''Root Mean Square Layer Normalization'''
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)
""",
        usefulness_score=0.91,
        source_paper_id="1910.07467",
        introduced_year=2019,
        hyperparameters={"eps": 1e-6},
        time_complexity="O(n * d)",
        space_complexity="O(d)",
        flops_formula="3 * n * d",  # Faster than LayerNorm
        is_parallelizable=True,
        is_causal=False,
        math_operations=["square", "mean", "rsqrt", "scale"],
    ),
    Component(
        name="RotaryEmbedding",
        description="Rotary Position Embedding (RoPE). Encodes position in rotation.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class RotaryEmbedding(nn.Module):
    '''Rotary Position Embedding (RoPE)'''
    def __init__(self, dim, base=10000, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x):
        seq_len = x.shape[1]
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        return self.apply_rotary(x, cos, sin)

    def apply_rotary(self, x, cos, sin):
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
""",
        usefulness_score=0.89,
        source_paper_id="2104.09864",
        introduced_year=2021,
        hyperparameters={"base": 10000, "dim": 64},
        time_complexity="O(n * d)",
        space_complexity="O(d)",
        flops_formula="4 * n * d",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["sin", "cos", "rotate", "multiply"],
    ),
    Component(
        name="SwiGLU",
        description="Swish-Gated Linear Unit. Better FFN activation from PaLM/LLaMA.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class SwiGLU(nn.Module):
    '''SwiGLU activation for FFN (LLaMA style)'''
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or int(d_model * 8 / 3)  # LLaMA ratio
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.W_gate(x))
        up = self.W_up(x)
        return self.W_down(gate * up)
""",
        usefulness_score=0.87,
        source_paper_id="2002.05202",
        introduced_year=2020,
        hyperparameters={"d_ff_multiplier": 2.67},  # 8/3 ratio for param match
        time_complexity="O(n * d * d_ff)",
        space_complexity="O(d * d_ff)",
        flops_formula="3 * n * d * d_ff",  # 3 linear layers
        is_parallelizable=True,
        is_causal=False,
        math_operations=["linear", "silu", "multiply", "linear"],
    ),
    Component(
        name="GroupedQueryAttention",
        description="GQA: Multiple query heads share key-value heads. Memory efficient.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class GroupedQueryAttention(nn.Module):
    '''Grouped Query Attention (Mistral/LLaMA 2 style)'''
    def __init__(self, d_model, n_heads=32, n_kv_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads

        self.W_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x, mask=None):
        B, L, D = x.shape

        q = self.W_q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat KV heads for each query group
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.W_o(out)
""",
        usefulness_score=0.86,
        source_paper_id="2305.13245",
        introduced_year=2023,
        hyperparameters={"n_heads": 32, "n_kv_heads": 8, "head_dim": 128},
        time_complexity="O(n^2 * d)",
        space_complexity="O(n^2 + n*d/g)",  # g = grouping factor
        flops_formula="2*n*d^2 + 2*n^2*d/g",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["linear", "matmul", "softmax", "matmul", "linear"],
    ),
    Component(
        name="MultiQueryAttention",
        description="MQA: Single key-value head shared across all query heads. Max memory efficiency.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class MultiQueryAttention(nn.Module):
    '''Multi-Query Attention (PaLM style): 1 KV head, N query heads'''
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, self.head_dim, bias=False)  # Single head
        self.W_v = nn.Linear(d_model, self.head_dim, bias=False)  # Single head
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, L, D = x.shape

        q = self.W_q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).unsqueeze(1)  # [B, 1, L, head_dim]
        v = self.W_v(x).unsqueeze(1)  # [B, 1, L, head_dim]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)
""",
        usefulness_score=0.84,
        source_paper_id="2204.02311",  # PaLM paper
        introduced_year=2022,
        hyperparameters={"n_heads": 8, "head_dim": 64},
        time_complexity="O(n^2 * d)",
        space_complexity="O(n^2 + n*d/h)",  # Much less KV cache
        flops_formula="n*d^2 + 2*n^2*d/h",  # Reduced KV computation
        is_parallelizable=True,
        is_causal=False,
        math_operations=["linear", "matmul", "softmax", "matmul", "linear"],
    ),
    Component(
        name="SlidingWindowAttention",
        description="Local attention with fixed window size. O(n*w) complexity for long sequences.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class SlidingWindowAttention(nn.Module):
    '''Sliding Window Attention (Mistral style)'''
    def __init__(self, d_model, n_heads=8, window_size=4096):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, L, D = x.shape

        q = self.W_q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Create sliding window mask
        window_mask = torch.ones(L, L, device=x.device, dtype=torch.bool)
        for i in range(L):
            start = max(0, i - self.window_size)
            window_mask[i, start:i+1] = False  # Can attend within window

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(window_mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)
""",
        usefulness_score=0.85,
        source_paper_id="2310.06825",  # Mistral paper
        introduced_year=2023,
        hyperparameters={"n_heads": 8, "window_size": 4096},
        time_complexity="O(n * w * d)",  # w = window size
        space_complexity="O(n * w)",
        flops_formula="4*n*d^2 + 2*n*w*d",
        is_parallelizable=True,
        is_causal=True,
        math_operations=["linear", "matmul", "softmax", "matmul", "linear"],
    ),
    Component(
        name="LinearAttention",
        description="O(n) attention using kernel trick. No softmax, uses feature maps.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class LinearAttention(nn.Module):
    '''Linear Attention with ELU feature map'''
    def __init__(self, d_model, n_heads=8, eps=1e-6):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.eps = eps

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def feature_map(self, x):
        '''ELU-based feature map for positive features'''
        return F.elu(x) + 1

    def forward(self, x, mask=None):
        B, L, D = x.shape

        q = self.W_q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply feature map (no softmax!)
        q = self.feature_map(q)
        k = self.feature_map(k)

        # Linear attention: O(n * d^2) instead of O(n^2 * d)
        # (Q @ K^T) @ V  =>  Q @ (K^T @ V)
        kv = k.transpose(-2, -1) @ v  # [B, H, d, d]
        qkv = q @ kv  # [B, H, L, d]

        # Normalize
        k_sum = k.sum(dim=-2, keepdim=True)  # [B, H, 1, d]
        normalizer = (q * k_sum).sum(dim=-1, keepdim=True) + self.eps
        out = qkv / normalizer

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)
""",
        usefulness_score=0.79,
        source_paper_id="2006.16236",  # Performers paper
        introduced_year=2020,
        hyperparameters={"n_heads": 8, "feature_map": "elu"},
        time_complexity="O(n * d^2)",  # Linear in sequence length!
        space_complexity="O(d^2)",
        flops_formula="4*n*d^2 + n*d^2",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["linear", "feature_map", "matmul", "normalize"],
    ),
    # =====================================================
    # EFFICIENCY COMPONENTS
    # =====================================================
    Component(
        name="LoRA",
        description="Low-Rank Adaptation. Efficient fine-tuning by decomposing weight updates into low-rank matrices.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class LoRA(nn.Module):
    '''Low-Rank Adaptation layer for efficient fine-tuning'''
    def __init__(self, d_model, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        # Decomposed weight update: W' = W + BA where B is d_model x rank, A is rank x d_model
        self.lora_A = nn.Parameter(torch.zeros(d_model, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, d_model))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Apply low-rank update: x @ (BA) * scaling
        return x + (x @ self.lora_A @ self.lora_B) * self.scaling
""",
        usefulness_score=0.88,
        source_paper_id="2106.09685",  # LoRA paper
        introduced_year=2021,
        hyperparameters={"rank": 8, "alpha": 16},
        time_complexity="O(n * d * r)",
        space_complexity="O(d * r)",
        flops_formula="2 * n * d * r",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["matmul", "scale", "add"],
    ),
    Component(
        name="MoERouter",
        description="Mixture of Experts router. Selects top-k experts for each token using learned gating.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, n_experts]", "dtype": "float32"},
        code="""
class MoERouter(nn.Module):
    '''Top-k router for Mixture of Experts'''
    def __init__(self, d_model, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x):
        # x: [B, L, D] -> logits: [B, L, E]
        logits = self.gate(x)
        # Get top-k experts per token
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        return top_k_weights, top_k_indices
""",
        usefulness_score=0.85,
        source_paper_id="2101.03961",  # Switch Transformer
        introduced_year=2021,
        hyperparameters={"n_experts": 8, "top_k": 2},
        time_complexity="O(n * d * E)",
        space_complexity="O(d * E)",
        flops_formula="n * d * E",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["linear", "topk", "softmax"],
    ),
    Component(
        name="MoELayer",
        description="Full Mixture of Experts layer. Routes tokens to selected experts and combines outputs.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class MoELayer(nn.Module):
    '''Mixture of Experts with top-k routing'''
    def __init__(self, d_model, d_ff, n_experts=8, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        # Each expert is a simple FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(n_experts)
        ])

    def forward(self, x):
        B, L, D = x.shape
        # Get routing weights
        logits = self.gate(x)  # [B, L, E]
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)  # [B, L, k]

        # Compute expert outputs (simplified - full impl uses scatter/gather)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (indices == i).any(dim=-1)  # [B, L]
            if mask.any():
                expert_out = expert(x)
                weight = (indices == i).float() * weights
                weight = weight.sum(dim=-1, keepdim=True)
                out = out + expert_out * weight
        return out
""",
        usefulness_score=0.87,
        source_paper_id="2401.04088",  # Mixtral paper
        introduced_year=2024,
        hyperparameters={"n_experts": 8, "top_k": 2, "d_ff": 14336},
        time_complexity="O(n * k * d * d_ff / E)",  # Only k experts active
        space_complexity="O(E * d * d_ff)",
        flops_formula="n * k * 2 * d * d_ff",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["linear", "topk", "softmax", "gelu", "weighted_sum"],
    ),
    Component(
        name="KVCache",
        description="Key-Value cache for efficient autoregressive generation. Stores computed KV pairs.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, total_len, d_model]", "dtype": "float32"},
        code="""
class KVCache(nn.Module):
    '''KV Cache for efficient inference'''
    def __init__(self, max_seq_len, n_heads, head_dim):
        super().__init__()
        self.max_seq_len = max_seq_len
        # Cached keys and values
        self.register_buffer('k_cache', None)
        self.register_buffer('v_cache', None)
        self.seq_len = 0

    def update(self, k, v):
        '''Add new k,v to cache and return full cached tensors'''
        if self.k_cache is None:
            self.k_cache = k
            self.v_cache = v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=2)
            self.v_cache = torch.cat([self.v_cache, v], dim=2)
        self.seq_len = self.k_cache.size(2)
        return self.k_cache, self.v_cache

    def forward(self, k, v):
        return self.update(k, v)
""",
        usefulness_score=0.91,
        source_paper_id="1706.03762",  # Original transformer, implicit
        introduced_year=2017,
        hyperparameters={"max_seq_len": 2048},
        time_complexity="O(1)",  # Just concatenation
        space_complexity="O(n * d)",
        flops_formula="0 (just memory)",
        is_parallelizable=True,
        is_causal=True,
        math_operations=["concat"],
    ),
    Component(
        name="ALiBi",
        description="Attention with Linear Biases. Adds linear position bias to attention scores, enabling length extrapolation.",
        interface_in={"shape": "[batch, n_heads, seq_len, seq_len]", "dtype": "float32"},
        interface_out={"shape": "[batch, n_heads, seq_len, seq_len]", "dtype": "float32"},
        code="""
class ALiBi(nn.Module):
    '''Attention with Linear Biases for position encoding'''
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
        # Slopes decrease geometrically: 2^(-8/n), 2^(-16/n), ...
        slopes = torch.tensor([2 ** (-8 * i / n_heads) for i in range(1, n_heads + 1)])
        self.register_buffer('slopes', slopes.view(1, n_heads, 1, 1))

    def forward(self, attn_scores, seq_len=None):
        if seq_len is None:
            seq_len = attn_scores.size(-1)
        # Create position bias matrix: slope * |i - j|
        positions = torch.arange(seq_len, device=attn_scores.device)
        bias = -torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        bias = bias.unsqueeze(0).unsqueeze(0) * self.slopes
        return attn_scores + bias
""",
        usefulness_score=0.86,
        source_paper_id="2108.12409",  # ALiBi paper
        introduced_year=2021,
        hyperparameters={"n_heads": 8},
        time_complexity="O(n^2)",
        space_complexity="O(n^2)",
        flops_formula="n^2 * h",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["subtract", "abs", "scale", "add"],
    ),
    # =====================================================
    # FOUNDATIONAL COMPONENTS (pre-2020)
    # =====================================================
    Component(
        name="CrossAttention",
        description="Encoder-decoder cross attention. Queries from decoder attend to encoder outputs (T5, BART).",
        interface_in={"shape": "[batch, tgt_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, tgt_len, d_model]", "dtype": "float32"},
        code="""
class CrossAttention(nn.Module):
    '''Encoder-decoder cross attention'''
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, encoder_out, mask=None):
        B, L, D = x.shape
        _, S, _ = encoder_out.shape  # Source length

        q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(encoder_out).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(encoder_out).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)
""",
        usefulness_score=0.89,
        source_paper_id="1706.03762",  # Original Transformer
        introduced_year=2017,
        hyperparameters={"n_heads": 8, "d_model": 512},
        time_complexity="O(n * m * d)",  # n=target, m=source
        space_complexity="O(n * m)",
        flops_formula="4*n*d^2 + 2*n*m*d",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["matmul", "scale", "softmax", "matmul", "linear"],
    ),
    Component(
        name="RelativePositionBias",
        description="T5-style learned relative position bias. Adds learned bias based on relative position to attention.",
        interface_in={"shape": "[batch, n_heads, seq_len, seq_len]", "dtype": "float32"},
        interface_out={"shape": "[batch, n_heads, seq_len, seq_len]", "dtype": "float32"},
        code="""
class RelativePositionBias(nn.Module):
    '''T5-style relative position bias'''
    def __init__(self, n_heads, num_buckets=32, max_distance=128):
        super().__init__()
        self.n_heads = n_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, n_heads)

    def _relative_position_bucket(self, relative_position):
        '''Bucket relative positions into num_buckets categories'''
        num_buckets = self.num_buckets // 2
        ret = (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(self.max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, attn_scores):
        seq_len = attn_scores.size(-1)
        positions = torch.arange(seq_len, device=attn_scores.device)
        relative_position = positions.unsqueeze(0) - positions.unsqueeze(1)
        buckets = self._relative_position_bucket(relative_position)
        bias = self.relative_attention_bias(buckets).permute(2, 0, 1).unsqueeze(0)
        return attn_scores + bias
""",
        usefulness_score=0.84,
        source_paper_id="1910.10683",  # T5 paper
        introduced_year=2019,
        hyperparameters={"num_buckets": 32, "max_distance": 128},
        time_complexity="O(n^2)",
        space_complexity="O(B * n_heads)",
        flops_formula="n^2 + n^2 * h",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["embedding", "bucket", "add"],
    ),
    Component(
        name="GatedLinearUnit",
        description="GLU activation. Splits input and uses one half to gate the other. Used in PaLM, GPT-J.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class GatedLinearUnit(nn.Module):
    '''Gated Linear Unit (GLU) for FFN'''
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff * 2)  # Project to 2x for gating
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden, gate = hidden.chunk(2, dim=-1)
        return self.fc2(hidden * torch.sigmoid(gate))
""",
        usefulness_score=0.83,
        source_paper_id="1612.08083",  # GLU paper
        introduced_year=2016,
        hyperparameters={"d_ff": 2048},
        time_complexity="O(n * d * d_ff)",
        space_complexity="O(d * d_ff)",
        flops_formula="n * (2 * d * d_ff + d_ff * d)",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["linear", "chunk", "sigmoid", "mul", "linear"],
    ),
    Component(
        name="GeGLU",
        description="GELU-gated GLU variant. Uses GELU instead of sigmoid for gating. Used in PaLM.",
        interface_in={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        interface_out={"shape": "[batch, seq_len, d_model]", "dtype": "float32"},
        code="""
class GeGLU(nn.Module):
    '''GELU-Gated Linear Unit'''
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff * 2)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden, gate = hidden.chunk(2, dim=-1)
        return self.fc2(hidden * F.gelu(gate))
""",
        usefulness_score=0.85,
        source_paper_id="2002.05202",  # GLU Variants paper
        introduced_year=2020,
        hyperparameters={"d_ff": 2048},
        time_complexity="O(n * d * d_ff)",
        space_complexity="O(d * d_ff)",
        flops_formula="n * (2 * d * d_ff + d_ff * d)",
        is_parallelizable=True,
        is_causal=False,
        math_operations=["linear", "chunk", "gelu", "mul", "linear"],
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
    # New architectures
    {
        "name": "T5",
        "description": "Text-to-Text Transfer Transformer. Encoder-decoder with relative position bias.",
        "paper_url": "https://arxiv.org/abs/1910.10683",
        "components": ["Embedding", "MultiHeadAttention", "CrossAttention", "RelativePositionBias", "FeedForward", "LayerNorm", "SoftmaxOutput"],
        "score": 0.94
    },
    {
        "name": "Mixtral-8x7B",
        "description": "Sparse MoE with 8 experts, 2 active per token. Based on Mistral with MoE layers.",
        "paper_url": "https://arxiv.org/abs/2401.04088",
        "components": ["Embedding", "GroupedQueryAttention", "MoELayer", "RMSNorm", "RotaryEmbedding", "CausalMask", "SoftmaxOutput"],
        "score": 0.94
    },
    {
        "name": "PaLM",
        "description": "Pathways Language Model. Parallel attention+FFN with SwiGLU and multi-query attention.",
        "paper_url": "https://arxiv.org/abs/2204.02311",
        "components": ["Embedding", "MultiQueryAttention", "SwiGLU", "RMSNorm", "RotaryEmbedding", "CausalMask", "SoftmaxOutput"],
        "score": 0.95
    },
    {
        "name": "BLOOM",
        "description": "BigScience Large Open-science Open-access Multilingual model with ALiBi positions.",
        "paper_url": "https://arxiv.org/abs/2211.05100",
        "components": ["Embedding", "MultiHeadAttention", "ALiBi", "FeedForward", "LayerNorm", "CausalMask", "SoftmaxOutput"],
        "score": 0.91
    },
    {
        "name": "Falcon",
        "description": "Technology Innovation Institute model with multi-query attention and ALiBi.",
        "paper_url": "https://arxiv.org/abs/2306.01116",
        "components": ["Embedding", "MultiQueryAttention", "ALiBi", "FeedForward", "LayerNorm", "CausalMask", "SoftmaxOutput"],
        "score": 0.92
    },
    {
        "name": "GPT-NeoX",
        "description": "EleutherAI's GPT variant with rotary embeddings and parallel attention.",
        "paper_url": "https://arxiv.org/abs/2204.06745",
        "components": ["Embedding", "MultiHeadAttention", "RotaryEmbedding", "FeedForward", "LayerNorm", "CausalMask", "SoftmaxOutput"],
        "score": 0.89
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
    # New attention variants
    ("MultiQueryAttention", "FeedForward", 0.92),
    ("MultiQueryAttention", "RMSNorm", 0.91),
    ("MultiQueryAttention", "LayerNorm", 0.90),
    ("SlidingWindowAttention", "FeedForward", 0.90),
    ("SlidingWindowAttention", "RMSNorm", 0.89),
    ("SlidingWindowAttention", "RotaryEmbedding", 0.88),
    ("LinearAttention", "FeedForward", 0.85),
    ("LinearAttention", "LayerNorm", 0.84),
    ("Embedding", "MultiQueryAttention", 0.88),
    ("Embedding", "SlidingWindowAttention", 0.87),
    ("Embedding", "LinearAttention", 0.83),
    # New efficiency/foundational component pairs
    ("LoRA", "MultiHeadAttention", 0.90),
    ("LoRA", "FeedForward", 0.89),
    ("LoRA", "GroupedQueryAttention", 0.88),
    ("MoELayer", "RMSNorm", 0.91),
    ("MoELayer", "LayerNorm", 0.90),
    ("MoELayer", "MultiHeadAttention", 0.87),
    ("MoELayer", "GroupedQueryAttention", 0.89),
    ("MoERouter", "FeedForward", 0.86),
    ("KVCache", "MultiHeadAttention", 0.92),
    ("KVCache", "GroupedQueryAttention", 0.93),
    ("KVCache", "MultiQueryAttention", 0.94),
    ("ALiBi", "MultiHeadAttention", 0.88),
    ("ALiBi", "MultiQueryAttention", 0.87),
    ("CrossAttention", "MultiHeadAttention", 0.91),
    ("CrossAttention", "LayerNorm", 0.90),
    ("CrossAttention", "FeedForward", 0.88),
    ("RelativePositionBias", "MultiHeadAttention", 0.89),
    ("RelativePositionBias", "CrossAttention", 0.90),
    ("GatedLinearUnit", "LayerNorm", 0.86),
    ("GatedLinearUnit", "RMSNorm", 0.87),
    ("GeGLU", "LayerNorm", 0.87),
    ("GeGLU", "RMSNorm", 0.88),
    ("SwiGLU", "GeGLU", 0.92),  # Similar components
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

        comp_ids = []
        missing = []
        for name in arch["components"]:
            if name in name_to_id:
                comp_ids.append(name_to_id[name])
            else:
                missing.append(name)
        if missing and verbose:
            print(f"  [WARN] {arch['name']}: missing components {missing}")

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
                    c2c_score=DEFAULT_SEED_RELATIONSHIP_SCORE
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
