#!/usr/bin/env python3
"""
Dream & Train Pipeline

Uses the composer to dream up architectures, then trains them on Modal GPU
and logs results to the database for empirical architecture search.

Run with: PYTHONUNBUFFERED=1 .venv-modal/bin/python scripts/dream_and_train.py

Pipeline:
1. Composer dreams component combinations (greedy, random, crossover, mutate)
2. Component patterns mapped to trainable code
3. Train on Modal A100
4. Results logged to training_runs + auto-generated insights
5. Patterns accumulate in DB for learning
"""
import sys
import json
import hashlib
import random
from pathlib import Path
from datetime import datetime

# Add both paths for proper imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "arcfusion"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cloud_train_fair import CONFIG, app, train_model, save_result_to_db, generate_auto_insight
from db import ArcFusionDB, DreamCandidate
from surrogate_model import SurrogateModel, ArchFeatures, extract_features, retrain_if_needed

# Import composer components we need (avoiding full package import)
# We'll use a simplified dreamer that queries the DB directly


# Component name patterns -> model template mappings
COMPONENT_PATTERNS = {
    # Attention mechanisms
    'multi-head attention': ('gqa', {'n_kv_heads': 8}),
    'scaled dot-product': ('gqa', {'n_kv_heads': 8}),
    'grouped query': ('gqa', {'n_kv_heads': 2}),
    'multi-query': ('gqa', {'n_kv_heads': 1}),
    'linear attention': ('linear', {}),

    # SSM/Mamba
    'selective ssm': ('mamba', {'use_parallel_scan': True}),
    'mamba': ('mamba', {'use_parallel_scan': True}),
    's4': ('mamba', {'use_parallel_scan': True}),

    # Position encodings
    'rotary': ('rope', {}),
    'sinusoidal': ('sinusoidal', {}),
    'learned position': ('learned_pos', {}),

    # Efficiency
    'flash attention': ('flash', {}),
}


def get_component_category(name: str) -> str:
    """Infer category from component name."""
    name_lower = name.lower()
    if any(x in name_lower for x in ['attention', 'query', 'key', 'value']):
        return 'attention'
    if any(x in name_lower for x in ['mamba', 'ssm', 's4', 'state space']):
        return 'ssm'
    if any(x in name_lower for x in ['position', 'rotary', 'rope', 'sinusoidal']):
        return 'position'
    if any(x in name_lower for x in ['norm', 'layer norm', 'rms']):
        return 'normalization'
    if any(x in name_lower for x in ['ffn', 'feed forward', 'mlp', 'gelu']):
        return 'ffn'
    if any(x in name_lower for x in ['embed', 'token']):
        return 'embedding'
    if any(x in name_lower for x in ['flash', 'efficient', 'sparse']):
        return 'efficiency'
    return 'other'


def components_to_architecture_code(components: list, n_layers: int = 10) -> tuple[str, str]:
    """Convert a list of components to trainable PyTorch code.

    Returns: (code, model_name)
    """
    # Analyze components to determine architecture type
    categories = {}
    for comp in components:
        cat = get_component_category(comp.name)
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(comp.name)

    # Build model name from components
    name_parts = []

    # Determine attention type
    attn_type = 'MHA'
    n_kv_heads = 8
    if 'ssm' in categories:
        attn_type = 'Mamba'
    elif 'attention' in categories:
        attn_names = ' '.join(categories['attention']).lower()
        if 'grouped query' in attn_names or 'gqa' in attn_names:
            attn_type = 'GQA'
            n_kv_heads = 2
        elif 'multi-query' in attn_names or 'mqa' in attn_names:
            attn_type = 'MQA'
            n_kv_heads = 1
        elif 'linear' in attn_names:
            attn_type = 'LinearAttn'

    name_parts.append(attn_type)

    # Add position encoding suffix
    if 'position' in categories:
        pos_names = ' '.join(categories['position']).lower()
        if 'rotary' in pos_names or 'rope' in pos_names:
            name_parts.append('RoPE')

    # Add efficiency suffix
    if 'efficiency' in categories:
        eff_names = ' '.join(categories['efficiency']).lower()
        if 'flash' in eff_names:
            name_parts.append('Flash')

    model_name = f"Dreamed_{'_'.join(name_parts)}_{n_layers}L"

    # Generate code based on detected architecture
    if attn_type == 'Mamba':
        code = generate_mamba_code(n_layers)
    elif attn_type == 'LinearAttn':
        code = generate_linear_attention_code(n_layers)
    else:
        code = generate_gqa_code(n_layers, n_kv_heads)

    return code, model_name


def generate_gqa_code(n_layers: int, n_kv_heads: int) -> str:
    """Generate GQA/MQA/MHA code."""
    return f'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
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
        self.attn = GroupedQueryAttention(d_model, n_heads)
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

class DreamedModel(nn.Module):
    def __init__(self, d_model, vocab_size, n_layers, n_heads):
        super().__init__()
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


def generate_mamba_code(n_layers: int) -> str:
    """Generate Mamba code with parallel scan."""
    return f'''
import torch
import torch.nn as nn
import torch.nn.functional as F

def parallel_scan(A, B_x):
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
    return A_cumsum_shifted * weighted_sum

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                 padding=d_conv - 1, groups=self.d_inner, bias=True)
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
        x_conv = self.conv1d(x_conv)[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        x_ssm = self.x_proj(x_conv)
        dt, B_input, C = x_ssm[:, :, :1], x_ssm[:, :, 1:self.d_state+1], x_ssm[:, :, self.d_state+1:]
        dt = F.softplus(self.dt_proj(dt))
        A_bar = torch.exp(dt.unsqueeze(-1) * self.A.unsqueeze(0).unsqueeze(0))
        B_bar = dt.unsqueeze(-1) * B_input.unsqueeze(2)
        B_x = B_bar * x_conv.unsqueeze(-1)
        h = parallel_scan(A_bar, B_x)
        y = (h * C.unsqueeze(2)).sum(-1) + self.D.unsqueeze(0).unsqueeze(0) * x_conv
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y))

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
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

class DreamedModel(nn.Module):
    def __init__(self, d_model, vocab_size, n_layers, n_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model) for _ in range({n_layers})])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))
'''


def generate_linear_attention_code(n_layers: int) -> str:
    """Generate Linear Attention code."""
    return f'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        # Linear attention: kernel feature map (ELU + 1)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        # Causal linear attention via cumsum
        kv = torch.einsum('bhnd,bhne->bhnde', k, v)
        kv_cumsum = torch.cumsum(kv, dim=2)
        k_cumsum = torch.cumsum(k, dim=2)
        qkv = torch.einsum('bhnd,bhnde->bhne', q, kv_cumsum)
        qk = torch.einsum('bhnd,bhnd->bhn', q, k_cumsum).unsqueeze(-1)
        out = qkv / (qk + 1e-6)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.dropout(self.out(out))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = LinearAttention(d_model, n_heads)
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

class DreamedModel(nn.Module):
    def __init__(self, d_model, vocab_size, n_layers, n_heads):
        super().__init__()
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


class SimpleComponent:
    """Simplified component for dreaming without full composer import."""
    def __init__(self, name: str, component_id: str = None):
        self.name = name
        self.component_id = component_id or name.lower().replace(' ', '_')


def dream_architecture(db: ArcFusionDB, strategy: str = 'greedy',
                       temperature: float = 0.3, max_components: int = 6) -> tuple[list, str]:
    """Dream up an architecture using the specified strategy.

    Simplified dreamer that queries DB directly without full composer import.
    Returns: (components, strategy_name)
    """
    # Get all components from DB
    rows = db.conn.execute(
        "SELECT component_id, name FROM components ORDER BY usefulness_score DESC"
    ).fetchall()

    if not rows:
        return [], strategy

    all_components = [SimpleComponent(name=r[1], component_id=r[0]) for r in rows]

    # Categorize components
    by_category = {}
    for comp in all_components:
        cat = get_component_category(comp.name)
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(comp)

    components = []

    if strategy == 'greedy':
        # Greedy: pick best from each category
        category_order = ['position', 'embedding', 'attention', 'ssm', 'normalization', 'ffn', 'efficiency']
        for cat in category_order:
            if cat in by_category and len(components) < max_components:
                # Pick best (first, since sorted by usefulness)
                if temperature > 0 and len(by_category[cat]) > 1:
                    # Add some randomness
                    top_k = min(3, len(by_category[cat]))
                    components.append(random.choice(by_category[cat][:top_k]))
                else:
                    components.append(by_category[cat][0])

    elif strategy == 'random':
        # Random walk: sample from all components
        weights = [1.0 / (i + 1) for i in range(len(all_components))]  # Favor higher scored
        for _ in range(max_components):
            if temperature > 0.5:
                comp = random.choice(all_components)
            else:
                comp = random.choices(all_components, weights=weights)[0]
            if comp not in components:
                components.append(comp)

    else:
        # Default: just take top components
        components = all_components[:max_components]

    return components, strategy


def get_architecture_hash(components: list) -> str:
    """Get a short hash for an architecture based on its components."""
    names = sorted([c.name for c in components])
    return hashlib.md5('|'.join(names).encode()).hexdigest()[:8]


def get_existing_candidate_hashes(db: ArcFusionDB) -> set:
    """Load all existing candidate component hashes from dream_candidates table."""
    rows = db.conn.execute(
        "SELECT components_json FROM dream_candidates"
    ).fetchall()

    hashes = set()
    for (components_json,) in rows:
        try:
            names = sorted(json.loads(components_json))
            h = hashlib.md5('|'.join(names).encode()).hexdigest()[:8]
            hashes.add(h)
        except (json.JSONDecodeError, TypeError):
            continue
    return hashes


def get_untrained_promising_candidates(db: ArcFusionDB, limit: int = 5) -> list:
    """Get candidates with good predicted PPL that were never trained.

    Returns list of (candidate_id, components_json, predicted_ppl, predicted_time)
    """
    rows = db.conn.execute("""
        SELECT candidate_id, components_json, predicted_ppl, predicted_time
        FROM dream_candidates
        WHERE was_trained = 0 AND predicted_ppl > 0
        ORDER BY predicted_ppl ASC
        LIMIT ?
    """, (limit,)).fetchall()
    return rows


def dreamed_to_arch_features(components: list, n_layers: int = 10) -> ArchFeatures:
    """Convert dreamed components to ArchFeatures for surrogate model prediction."""
    categories = {}
    for comp in components:
        cat = get_component_category(comp.name)
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(comp.name.lower())

    # Determine attention type and KV heads
    n_kv_heads = 8  # Default MHA
    has_mamba = 'ssm' in categories
    has_linear_attn = False
    is_hybrid = False

    if 'attention' in categories:
        attn_names = ' '.join(categories['attention'])
        if 'grouped query' in attn_names or 'gqa' in attn_names:
            n_kv_heads = 2
        elif 'multi-query' in attn_names or 'mqa' in attn_names:
            n_kv_heads = 1
        elif 'linear' in attn_names:
            has_linear_attn = True

    # Check for hybrid (both SSM and attention)
    if has_mamba and 'attention' in categories:
        is_hybrid = True

    if has_mamba:
        n_kv_heads = 0  # Mamba doesn't use KV

    return ArchFeatures(
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        has_mamba=has_mamba,
        has_linear_attn=has_linear_attn,
        is_hybrid=is_hybrid,
        is_fast_mamba=has_mamba,  # Assume parallel scan
        d_model=256,
        n_heads=8,
    )


def screen_candidates_with_surrogate(candidates: list, n_layers: int,
                                      surrogate: SurrogateModel, top_k: int = 3) -> tuple[list, list]:
    """Screen candidates using surrogate model.

    Returns: (top_k_candidates, all_candidates_with_predictions)
    """
    import numpy as np

    if not candidates:
        return [], []

    # Convert all candidates to features and predict
    features_list = []
    for components, strategy, temperature in candidates:
        features = dreamed_to_arch_features(components, n_layers)
        features_list.append((components, strategy, temperature, features))

    # Predict PPL and time for all
    X = np.array([f[3].to_vector() for f in features_list])
    pred_ppl = surrogate.predict_ppl(X)
    pred_time = surrogate.predict_time(X) if surrogate.weights_time is not None else np.zeros(len(X))

    # All candidates with predictions
    all_with_preds = [(f[0], f[1], f[2], f[3], ppl, time)
                       for f, ppl, time in zip(features_list, pred_ppl, pred_time)]

    # Rank by predicted PPL (lower is better)
    ranked = sorted(all_with_preds, key=lambda x: x[4])

    # Return top_k and all
    top_k_cands = [(c[0], c[1], c[2], c[4]) for c in ranked[:top_k]]  # components, strategy, temp, pred_ppl
    return top_k_cands, all_with_preds


def save_all_candidates_to_db(db: ArcFusionDB, candidates_with_preds: list,
                               n_layers: int, selected_indices: set) -> dict:
    """Save all dream candidates to the database.

    Args:
        db: Database connection
        candidates_with_preds: List of (components, strategy, temp, features, pred_ppl, pred_time)
        n_layers: Number of layers
        selected_indices: Set of indices that were selected for training

    Returns:
        Dict mapping (strategy, temp) -> candidate_id for lookup
    """
    candidate_map = {}

    for i, (components, strategy, temp, features, pred_ppl, pred_time) in enumerate(candidates_with_preds):
        # Determine architecture type from features
        if features.has_mamba:
            arch_type = 'mamba'
        elif features.has_linear_attn:
            arch_type = 'linear'
        elif features.n_kv_heads == 1:
            arch_type = 'mqa'
        elif features.n_kv_heads < features.n_heads:
            arch_type = 'gqa'
        else:
            arch_type = 'mha'

        # Create candidate
        candidate = DreamCandidate(
            strategy=strategy,
            temperature=temp,
            components_json=json.dumps([c.name for c in components]),
            n_layers=n_layers,
            n_kv_heads=features.n_kv_heads,
            has_mamba=features.has_mamba,
            has_linear_attn=features.has_linear_attn,
            is_hybrid=features.is_hybrid,
            arch_type=arch_type,
            predicted_ppl=pred_ppl,
            predicted_time=pred_time,
            was_trained=False,  # Will update later if trained
            notes=f"selected_for_training={i in selected_indices}"
        )

        try:
            db.add_dream_candidate(candidate)
            candidate_map[(strategy, temp)] = candidate.candidate_id
        except Exception as e:
            # Handle duplicate (same components dreamed before)
            print(f"  Note: Candidate may already exist: {e}")

    return candidate_map


def main():
    print("=" * 70)
    print("DREAM & TRAIN PIPELINE (with Surrogate Screening)")
    print("=" * 70)
    print("Dreaming architectures, screening with surrogate model, training best on GPU")
    print()
    sys.stdout.flush()

    # Connect to DB
    db_path = Path(__file__).parent.parent / "arcfusion.db"
    db = ArcFusionDB(str(db_path))

    # Load surrogate model for screening
    surrogate_path = Path(__file__).parent.parent / "surrogate_model.pkl"
    surrogate = SurrogateModel()
    use_surrogate = False
    if surrogate_path.exists():
        try:
            surrogate.load(str(surrogate_path))
            use_surrogate = True
            print(f"Loaded surrogate model for candidate screening")
        except Exception as e:
            print(f"Warning: Could not load surrogate model: {e}")
    else:
        print("No surrogate model found, training without pre-screening")

    # Configuration
    n_candidates = 20 if use_surrogate else 3  # Dream many if screening
    n_to_train = 3  # Only train top 3
    n_layers = 14  # Use 14 layers (good balance)
    strategies = ['greedy', 'random']  # Alternate strategies

    # Load existing candidate hashes to avoid re-dreaming
    existing_hashes = get_existing_candidate_hashes(db)
    print(f"Found {len(existing_hashes)} previously dreamed architectures")

    # Check for promising untrained candidates
    untrained = get_untrained_promising_candidates(db, limit=n_to_train)
    if untrained:
        print(f"Found {len(untrained)} promising untrained candidates:")
        for cid, comp_json, pred_ppl, pred_time in untrained:
            print(f"  - {cid}: pred_ppl={pred_ppl:.1f}")

    # Get baseline for comparison
    baseline_runs = db.list_training_runs(model_name="Transformer_MHA", success_only=True, limit=1)
    if not baseline_runs:
        baseline_runs = db.list_training_runs(success_only=True, limit=1)
    baseline_ppl = baseline_runs[0].perplexity if baseline_runs else 280.0
    baseline_run_id = baseline_runs[0].run_id if baseline_runs else ""

    print(f"Baseline PPL: {baseline_ppl:.1f}")
    print(f"Will dream {n_candidates} candidates, train top {n_to_train}")
    print()
    sys.stdout.flush()

    # Phase 1: Dream many candidates (skipping already-dreamed ones)
    print("=" * 70)
    print("PHASE 1: Dreaming candidate architectures")
    print("=" * 70)

    candidates = []
    skipped = 0
    max_attempts = n_candidates * 3  # Try more to find novel ones
    attempt = 0
    session_hashes = set()  # Track hashes within this session too

    while len(candidates) < n_candidates and attempt < max_attempts:
        strategy = strategies[attempt % len(strategies)]
        temperature = 0.2 + (attempt * 0.03)  # Vary exploration more finely

        components, _ = dream_architecture(db, strategy=strategy, temperature=temperature)
        if components:
            arch_hash = get_architecture_hash(components)

            # Skip if already in DB or already dreamed this session
            if arch_hash in existing_hashes or arch_hash in session_hashes:
                skipped += 1
                attempt += 1
                continue

            session_hashes.add(arch_hash)
            candidates.append((components, strategy, temperature))
            print(f"  [{len(candidates)}/{n_candidates}] {strategy} (t={temperature:.2f}): {len(components)} components")

        attempt += 1

    print(f"\nDreamed {len(candidates)} novel candidate architectures (skipped {skipped} duplicates)")
    sys.stdout.flush()

    # Phase 2: Screen with surrogate model and save all candidates to DB
    candidate_map = {}  # Maps (strategy, temp) -> candidate_id
    all_candidates_with_preds = []

    if use_surrogate and len(candidates) > n_to_train:
        print(f"\n{'=' * 70}")
        print("PHASE 2: Screening with surrogate model")
        print("=" * 70)

        screened, all_candidates_with_preds = screen_candidates_with_surrogate(
            candidates, n_layers, surrogate, top_k=n_to_train)

        print(f"\nTop {len(screened)} candidates by predicted PPL:")
        for i, (components, strategy, temp, pred_ppl) in enumerate(screened):
            comp_names = [c.name for c in components[:3]]
            print(f"  {i+1}. {strategy} (t={temp:.2f}): pred={pred_ppl:.1f} PPL - {comp_names}...")

        # Save ALL candidates to DB (with predictions)
        selected_indices = set(range(n_to_train))  # Top k are selected
        print(f"\nSaving {len(all_candidates_with_preds)} candidates to dream_candidates table...")
        candidate_map = save_all_candidates_to_db(db, all_candidates_with_preds, n_layers, selected_indices)
        print(f"  Saved {len(candidate_map)} candidates")

        # Convert to format for training
        to_train = [(c, s, t) for c, s, t, _ in screened]
    else:
        to_train = candidates[:n_to_train]

    print()
    sys.stdout.flush()

    # Phase 3: Train selected architectures
    print("=" * 70)
    print(f"PHASE 3: Training {len(to_train)} selected architectures")
    print("=" * 70)

    results = []

    for i, (components, strategy, temperature) in enumerate(to_train):
        print(f"\n{'-' * 70}")
        print(f"TRAIN {i+1}/{len(to_train)}: Strategy={strategy}, Temperature={temperature:.2f}")
        print("-" * 70)
        sys.stdout.flush()

        if not components:
            print("No components dreamed, skipping...")
            continue

        print(f"Dreamed {len(components)} components:")
        for c in components:
            print(f"  - {c.name}")
        sys.stdout.flush()

        # Convert to trainable code
        code, model_name = components_to_architecture_code(components, n_layers=n_layers)
        arch_hash = get_architecture_hash(components)
        model_name = f"{model_name}_{arch_hash}"

        # Check if already trained
        existing = db.list_training_runs(model_name=model_name, success_only=True, limit=1)
        if existing:
            print(f"SKIP: {model_name} already trained ({existing[0].perplexity:.1f} PPL)")
            continue

        print(f"\nTraining: {model_name}")
        sys.stdout.flush()

        # Train on Modal
        try:
            with app.run():
                result = train_model.remote(code, "DreamedModel", CONFIG)

            if result["success"]:
                ppl = result["perplexity"]
                time_s = result["time_seconds"]
                ppl_diff = ((ppl - baseline_ppl) / baseline_ppl) * 100

                print(f"  PPL: {ppl:.1f} ({ppl_diff:+.1f}% vs baseline)")
                print(f"  Time: {time_s:.1f}s")

                # Save to DB with dreamed model name
                result["model_name"] = model_name
                run_id = save_result_to_db(db, result, CONFIG, baseline_run_id, model_code=code)
                print(f"  Saved: {run_id}")

                # Log insight about the dream
                component_names = [c.name for c in components]
                insight_data = {
                    "strategy": strategy,
                    "temperature": temperature,
                    "components": component_names,
                    "n_layers": n_layers,
                    "ppl": ppl,
                    "time_seconds": time_s,
                    "vs_baseline_pct": ppl_diff
                }

                db.conn.execute("""
                    INSERT INTO training_insights
                    (insight_id, category, title, description, source_run_id, evidence_json, confidence, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    f"dream-{arch_hash}",
                    "dreamed_architecture",
                    f"Dreamed: {model_name}",
                    f"Architecture dreamed with {strategy} strategy (temp={temperature}). "
                    f"Components: {', '.join(component_names)}. "
                    f"Result: {ppl:.1f} PPL ({ppl_diff:+.1f}% vs baseline).",
                    run_id,
                    json.dumps(insight_data),
                    0.7,
                    f"dream,{strategy},{'-'.join([get_component_category(c.name) for c in components])}"
                ))
                db.conn.commit()

                # Update dream_candidate with actual training results
                candidate_key = (strategy, temperature)
                if candidate_key in candidate_map:
                    candidate_id = candidate_map[candidate_key]
                    db.update_dream_candidate_training(
                        candidate_id=candidate_id,
                        training_run_id=run_id,
                        actual_ppl=ppl,
                        actual_time=time_s
                    )
                    print(f"  Updated dream_candidate {candidate_id} with actual results")

                results.append({
                    "model_name": model_name,
                    "ppl": ppl,
                    "time": time_s,
                    "components": component_names,
                    "strategy": strategy
                })
            else:
                print(f"  FAILED: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"  ERROR: {e}")

        sys.stdout.flush()

    # Summary
    print(f"\n{'=' * 70}")
    print("DREAM & TRAIN SUMMARY")
    print("=" * 70)

    if results:
        results.sort(key=lambda x: x["ppl"])
        print(f"\nTrained {len(results)} architectures:\n")
        for r in results:
            print(f"  {r['model_name']}: {r['ppl']:.1f} PPL ({r['time']:.0f}s)")
            print(f"    Components: {', '.join(r['components'][:3])}...")

        best = results[0]
        print(f"\nBest: {best['model_name']} ({best['ppl']:.1f} PPL)")
    else:
        print("No architectures trained successfully")

    # Auto-update surrogate model if enough new data
    if results:
        print(f"\n{'=' * 70}")
        print("PHASE 4: Updating surrogate model")
        print("=" * 70)
        retrained, msg = retrain_if_needed(db, str(surrogate_path), min_new_samples=2)
        if retrained:
            print(f"  ✓ {msg}")
        else:
            print(f"  Skipped: {msg}")

    print()


if __name__ == "__main__":
    main()
