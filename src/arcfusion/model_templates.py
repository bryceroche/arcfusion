"""
Model Templates for Architecture Search

Reusable code generation templates for training experiments.
These generate PyTorch model code strings that can be sent to Modal for training.

Usage:
    from arcfusion.model_templates import generate_gqa_model, generate_mamba_model

    code, name = generate_gqa_model(n_layers=14, n_kv_heads=2)
    # Returns: (model_code_string, "Transformer_GQA14")
"""


def generate_gqa_model(n_layers: int, n_kv_heads: int, n_heads: int = 8) -> tuple[str, str]:
    """Generate Grouped Query Attention model with variable KV heads and layers.

    Args:
        n_layers: Number of transformer layers (e.g., 10, 14, 18, 32)
        n_kv_heads: Number of key-value heads
            - 1: MQA (Multi-Query Attention) - most memory efficient
            - 2: GQA with 2 KV heads
            - 4: GQA with 4 KV heads
            - 8: Standard MHA (Multi-Head Attention)
        n_heads: Number of query heads (default 8)

    Returns:
        tuple: (model_code_string, model_class_name)

    Example:
        >>> code, name = generate_gqa_model(14, 2)
        >>> name
        'Transformer_GQA14'
        >>> 'class Transformer_GQA14' in code
        True
    """
    # Name based on kv_heads: 1=MQA, 2=GQA, 4=GQA4, 8=MHA
    if n_kv_heads == 1:
        attn_name = "MQA"
    elif n_kv_heads == 2:
        attn_name = "GQA"
    elif n_kv_heads == 4:
        attn_name = "GQA4"
    elif n_kv_heads == n_heads:
        attn_name = "MHA"
    else:
        attn_name = f"KV{n_kv_heads}"

    model_name = f"Transformer_{attn_name}{n_layers}"

    code = f'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with {n_kv_heads} KV heads."""
    def __init__(self, d_model, n_heads, n_kv_heads={n_kv_heads}, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(x).reshape(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(x).reshape(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # Repeat KV heads to match Q heads
        n_rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(n_rep, dim=1).contiguous()
        v = v.repeat_interleave(n_rep, dim=1).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads={n_kv_heads})
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class {model_name}(nn.Module):
    """Transformer with {attn_name} attention ({n_kv_heads} KV heads) and {n_layers} layers."""
    def __init__(self, d_model, vocab_size, n_layers, n_heads):
        super().__init__()
        # Note: n_layers param ignored - layer count baked in at code generation
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(2048, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range({n_layers})])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, N = x.shape
        x = self.embed(x) + self.pos(torch.arange(N, device=x.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))
'''
    return code, model_name


def generate_mamba_model(n_layers: int, d_state: int = 16, use_parallel_scan: bool = True) -> tuple[str, str]:
    """Generate Mamba SSM model with variable layers and state dimension.

    Args:
        n_layers: Number of Mamba blocks
        d_state: State space dimension (default 16)
        use_parallel_scan: Use parallel cumsum/cumprod (4x faster) vs sequential

    Returns:
        tuple: (model_code_string, model_class_name)
    """
    suffix = "Fast" if use_parallel_scan else ""
    model_name = f"Transformer_Mamba{suffix}{n_layers}" if n_layers != 4 else f"Transformer_Mamba{suffix}"

    if use_parallel_scan:
        scan_code = '''
def parallel_scan(A, B_x):
    """Parallel associative scan for SSM - O(L) work, O(1) depth."""
    B, L, D, N = A.shape
    log_A = torch.log(A.clamp(min=1e-6))
    log_A_cumsum = torch.cumsum(log_A, dim=1)
    A_cumsum = torch.exp(log_A_cumsum)
    A_cumsum_shifted = torch.cat([
        torch.ones(B, 1, D, N, device=A.device, dtype=A.dtype),
        A_cumsum[:, :-1]
    ], dim=1)
    weighted_inputs = B_x / (A_cumsum_shifted + 1e-6)
    weighted_sum = torch.cumsum(weighted_inputs, dim=1)
    h = A_cumsum_shifted * weighted_sum
    return h
'''
        ssm_forward = '''
        # FAST: Parallel scan instead of sequential loop
        h = parallel_scan(A_bar, B_x)  # B, L, d_inner, d_state
'''
    else:
        scan_code = ''
        ssm_forward = '''
        # Sequential scan (slow but reference implementation)
        h = torch.zeros(B, L, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        h_prev = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        for t in range(L):
            h_prev = A_bar[:, t] * h_prev + B_x[:, t]
            h[:, t] = h_prev
'''

    code = f'''
import torch
import torch.nn as nn
import torch.nn.functional as F

{scan_code}

class SelectiveSSM(nn.Module):
    """Selective State Space Model{"" if not use_parallel_scan else " with parallel scan"}."""
    def __init__(self, d_model, d_state={d_state}, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            self.A.copy_(-torch.exp(torch.linspace(0, 4, d_state)).unsqueeze(0).expand(self.d_inner, -1))

    def forward(self, x):
        B, L, _ = x.shape

        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)

        x_conv = x_main.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_ssm = self.x_proj(x_conv)
        dt, B_input, C = x_ssm[:, :, :1], x_ssm[:, :, 1:self.d_state+1], x_ssm[:, :, self.d_state+1:]
        dt = F.softplus(self.dt_proj(dt))

        A = self.A
        A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        B_bar = dt.unsqueeze(-1) * B_input.unsqueeze(2)
        B_x = B_bar * x_conv.unsqueeze(-1)
{ssm_forward}
        y = (h * C.unsqueeze(2)).sum(-1) + self.D.unsqueeze(0).unsqueeze(0) * x_conv
        y = y * F.silu(z)
        y = self.out_proj(y)
        return self.dropout(y)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state={d_state}):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = x + self.ssm(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class {model_name}(nn.Module):
    """Mamba with {n_layers} layers{"" if not use_parallel_scan else " (parallel scan)"}."""
    def __init__(self, d_model, vocab_size, n_layers, n_heads):
        super().__init__()
        # n_heads unused - Mamba doesn't use attention, kept for API compatibility
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state={d_state}) for _ in range({n_layers})])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, N = x.shape
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))
'''
    return code, model_name


# Convenience functions for common configurations
def mqa(n_layers: int) -> tuple[str, str]:
    """Multi-Query Attention (1 KV head)."""
    return generate_gqa_model(n_layers, n_kv_heads=1)

def gqa(n_layers: int, n_kv_heads: int = 2) -> tuple[str, str]:
    """Grouped Query Attention."""
    return generate_gqa_model(n_layers, n_kv_heads=n_kv_heads)

def mha(n_layers: int) -> tuple[str, str]:
    """Multi-Head Attention (8 KV heads = 8 Q heads)."""
    return generate_gqa_model(n_layers, n_kv_heads=8)

def mamba(n_layers: int, fast: bool = True) -> tuple[str, str]:
    """Mamba SSM."""
    return generate_mamba_model(n_layers, use_parallel_scan=fast)
