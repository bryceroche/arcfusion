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
