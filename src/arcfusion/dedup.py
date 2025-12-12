"""
Fuzzy deduplication for components and engines.

Identifies and merges duplicate entries based on:
- Normalized name matching
- Semantic similarity (key architectural terms)
- Configurable similarity thresholds
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from .db import ArcFusionDB, Component, Engine

# Similarity scoring weights
WEIGHT_SAME_NAME = 0.6       # Exact normalized name match
WEIGHT_SUBSTRING = 0.4       # One name is substring of other
WEIGHT_SEMANTIC_HIGH = 0.3   # >80% semantic term overlap
WEIGHT_SEMANTIC_MOD = 0.15   # >50% semantic term overlap
WEIGHT_SAME_PAPER = 0.1      # Same source paper

# Semantic overlap thresholds
SEMANTIC_OVERLAP_HIGH = 0.8
SEMANTIC_OVERLAP_MODERATE = 0.5

# Default deduplication threshold
DEFAULT_DEDUP_THRESHOLD = 0.5


@dataclass
class DuplicateGroup:
    """A group of potentially duplicate components."""
    canonical: Component  # The one to keep
    duplicates: list[Component]  # Others to merge/delete
    similarity_reason: str  # Why they're grouped


def _canonical_sort_key(c: Component) -> tuple:
    """Sort key for selecting canonical component: has code, higher score, shorter name."""
    return (
        -int(bool(c.code.strip())),  # Has code first
        -c.usefulness_score,          # Higher score
        len(c.name)                   # Shorter name
    )


def normalize_component_name(name: str, preserve_prefix: bool = True) -> str:
    """
    Normalize a component name for comparison.

    Args:
        name: Component name
        preserve_prefix: If True, keep architecture prefixes like BERT, LLaMA

    Examples:
        "RMSNorm (Root Mean Square Layer Normalization)" -> "rmsnorm"
        "Multi-Head Attention" -> "multiheadattention"
        "Position-wise Feed-Forward Networks" -> "positionwisefeedforwardnetwork"
    """
    # Remove parenthetical content
    name = re.sub(r'\([^)]*\)', '', name)
    # Lowercase
    name = name.lower()
    # Remove special characters and spaces
    name = re.sub(r'[^a-z0-9]', '', name)
    # Handle common variations
    name = name.replace('networks', 'network')
    name = name.replace('connections', 'connection')
    name = name.replace('embeddings', 'embedding')
    return name


# Architecture-specific prefixes that indicate variants (should NOT be merged)
ARCHITECTURE_PREFIXES = [
    'bert', 'gpt', 'llama', 'mistral', 'transformer', 'mamba', 'rwkv',
    'retnet', 'flash', 'blocksparse', 'sparse', 'linear', 'efficient',
    'masked', 'causal', 'bidirectional', 'cross', 'self', 'rope', 'rotary',
    'grouped', 'multi-query', 'sliding', 'local', 'global', 'axial'
]


def is_architecture_variant(name1: str, name2: str) -> bool:
    """
    Check if two names represent architecture-specific variants that should stay separate.

    Returns True if they should NOT be merged (one is a variant of the other).
    """
    n1 = name1.lower()
    n2 = name2.lower()

    # Check if one has an architecture prefix the other doesn't
    for prefix in ARCHITECTURE_PREFIXES:
        n1_has = n1.startswith(prefix) or f' {prefix}' in f' {n1}'
        n2_has = n2.startswith(prefix) or f' {prefix}' in f' {n2}'

        # If one has the prefix and one doesn't, or they have different prefixes
        if n1_has != n2_has:
            return True

    # Check for "Block-Sparse" vs regular (specific variant indicators)
    variant_indicators = ['block-sparse', 'blocksparse', 'sparse', 'efficient', 'fast', 'linear']
    for indicator in variant_indicators:
        n1_has = indicator in n1.replace('-', '').replace(' ', '')
        n2_has = indicator in n2.replace('-', '').replace(' ', '')
        if n1_has != n2_has:
            return True

    return False


def extract_semantic_signature(name: str) -> frozenset:
    """
    Extract key architectural terms from a component name.
    Used for semantic grouping.
    """
    name_lower = name.lower()
    terms = set()

    # Core architectural patterns
    patterns = {
        'attention': ['attention', 'attn'],
        'multihead': ['multi-head', 'multihead', 'mha'],
        'transformer': ['transformer'],
        'encoder': ['encoder'],
        'decoder': ['decoder'],
        'embedding': ['embedding', 'embed'],
        'normalization': ['normalization', 'layernorm', 'rmsnorm', 'norm'],
        'feedforward': ['feed-forward', 'feedforward', 'ffn', 'mlp'],
        'positional': ['positional', 'position', 'pos'],
        'residual': ['residual', 'skip'],
        'dropout': ['dropout'],
        'activation': ['activation', 'gelu', 'relu', 'silu', 'swiglu', 'swish'],
        'softmax': ['softmax'],
        'ssm': ['ssm', 'state space', 'mamba', 'selective'],
        'rotary': ['rotary', 'rope'],
        'causal': ['causal', 'masked', 'autoregressive'],
        'cross': ['cross', 'encoder-decoder'],
        'self': ['self-attention', 'self attention'],
        'kvcache': ['kv-cache', 'kv cache', 'key-value cache'],
        'head': ['head'],
        'layer': ['layer'],
        'block': ['block'],
        'stack': ['stack'],
    }

    for term, variants in patterns.items():
        for variant in variants:
            if variant in name_lower:
                terms.add(term)
                break

    return frozenset(terms)


def calculate_similarity(comp1: Component, comp2: Component) -> tuple[float, str]:
    """
    Calculate similarity score between two components.
    Returns (score, reason) where score is 0-1.
    """
    reasons = []
    score = 0.0

    # Normalized name match (high weight)
    norm1 = normalize_component_name(comp1.name)
    norm2 = normalize_component_name(comp2.name)

    if norm1 == norm2:
        score += WEIGHT_SAME_NAME
        reasons.append("same normalized name")
    elif norm1 in norm2 or norm2 in norm1:
        score += WEIGHT_SUBSTRING
        reasons.append("substring match")

    # Semantic signature overlap
    sig1 = extract_semantic_signature(comp1.name)
    sig2 = extract_semantic_signature(comp2.name)

    if sig1 and sig2:
        overlap = len(sig1 & sig2) / max(len(sig1 | sig2), 1)
        if overlap > SEMANTIC_OVERLAP_HIGH:
            score += WEIGHT_SEMANTIC_HIGH
            reasons.append(f"high semantic overlap ({overlap:.0%})")
        elif overlap > SEMANTIC_OVERLAP_MODERATE:
            score += WEIGHT_SEMANTIC_MOD
            reasons.append(f"moderate semantic overlap ({overlap:.0%})")

    # Same source paper
    if comp1.source_paper_id and comp1.source_paper_id == comp2.source_paper_id:
        score += WEIGHT_SAME_PAPER
        reasons.append("same source paper")

    return min(score, 1.0), "; ".join(reasons) if reasons else "no match"


class ComponentDeduplicator:
    """Finds and merges duplicate components."""

    def __init__(self, db: ArcFusionDB):
        self.db = db

    def find_duplicates(self, threshold: float = DEFAULT_DEDUP_THRESHOLD) -> list[DuplicateGroup]:
        """
        Find groups of duplicate components.

        Args:
            threshold: Minimum similarity score to consider duplicates (0-1)

        Returns:
            List of DuplicateGroup objects
        """
        components = self.db.find_components()

        # Group by normalized name first (fast)
        norm_groups = defaultdict(list)
        for comp in components:
            key = normalize_component_name(comp.name)
            norm_groups[key].append(comp)

        duplicate_groups = []
        processed_ids = set()

        for key, group in norm_groups.items():
            if len(group) > 1:
                # Filter out architecture-specific variants
                # Only keep components that are truly duplicates, not variants
                filtered_group = []
                for comp in group:
                    is_variant = False
                    for other in group:
                        if comp.component_id != other.component_id:
                            if is_architecture_variant(comp.name, other.name):
                                is_variant = True
                                break
                    if not is_variant:
                        filtered_group.append(comp)

                # If filtering removed variants, use filtered; otherwise use original
                # (if all are variants of each other, none will be filtered)
                if len(filtered_group) > 1:
                    group = filtered_group
                elif len(filtered_group) == 0:
                    # All pairs are variants - skip this group
                    continue

                # Sort by: has code > higher score > shorter name (canonical)
                group.sort(key=_canonical_sort_key)

                canonical = group[0]
                duplicates = group[1:]

                if duplicates and canonical.component_id not in processed_ids:
                    duplicate_groups.append(DuplicateGroup(
                        canonical=canonical,
                        duplicates=duplicates,
                        similarity_reason="same normalized name"
                    ))
                    processed_ids.add(canonical.component_id)
                    for d in duplicates:
                        processed_ids.add(d.component_id)

        # Also check for semantic similarity (slower, more thorough)
        remaining = [c for c in components if c.component_id not in processed_ids]

        for i, comp1 in enumerate(remaining):
            if comp1.component_id in processed_ids:
                continue

            similar = []
            for comp2 in remaining[i+1:]:
                if comp2.component_id in processed_ids:
                    continue

                # Skip architecture-specific variants
                if is_architecture_variant(comp1.name, comp2.name):
                    continue

                score, reason = calculate_similarity(comp1, comp2)
                if score >= threshold:
                    similar.append((comp2, score, reason))

            if similar:
                # Sort by score descending
                similar.sort(key=lambda x: -x[1])

                # Determine canonical (prefer one with code, higher score)
                all_comps = [comp1] + [s[0] for s in similar]
                all_comps.sort(key=_canonical_sort_key)

                canonical = all_comps[0]
                duplicates = all_comps[1:]

                duplicate_groups.append(DuplicateGroup(
                    canonical=canonical,
                    duplicates=duplicates,
                    similarity_reason=similar[0][2]  # Use first match reason
                ))

                processed_ids.add(canonical.component_id)
                for d in duplicates:
                    processed_ids.add(d.component_id)

        return duplicate_groups

    def merge_group(self, group: DuplicateGroup, dry_run: bool = True) -> dict:
        """
        Merge a duplicate group, keeping the canonical and deleting others.

        Args:
            group: The duplicate group to merge
            dry_run: If True, don't actually modify the database

        Returns:
            Dict with merge statistics
        """
        result = {
            'canonical': group.canonical.name,
            'merged': [d.name for d in group.duplicates],
            'relationships_transferred': 0,
            'deleted': 0,
        }

        if dry_run:
            return result

        canonical_id = group.canonical.component_id

        for dup in group.duplicates:
            dup_id = dup.component_id

            # First, delete any relationships involving the duplicate that would
            # create conflicts when transferred to canonical
            # (i.e., relationships where canonical already has a relationship with the same component2)
            self.db.conn.execute(
                """DELETE FROM component_relationships
                   WHERE component1_id = ? AND component2_id IN (
                       SELECT component2_id FROM component_relationships WHERE component1_id = ?
                   )""",
                (dup_id, canonical_id)
            )
            self.db.conn.execute(
                """DELETE FROM component_relationships
                   WHERE component2_id = ? AND component1_id IN (
                       SELECT component1_id FROM component_relationships WHERE component2_id = ?
                   )""",
                (dup_id, canonical_id)
            )

            # Transfer remaining relationships to canonical
            # Update component1_id references
            cursor = self.db.conn.execute(
                """UPDATE component_relationships
                   SET component1_id = ?
                   WHERE component1_id = ? AND component2_id != ?""",
                (canonical_id, dup_id, canonical_id)
            )
            result['relationships_transferred'] += cursor.rowcount

            # Update component2_id references
            cursor = self.db.conn.execute(
                """UPDATE component_relationships
                   SET component2_id = ?
                   WHERE component2_id = ? AND component1_id != ?""",
                (canonical_id, dup_id, canonical_id)
            )
            result['relationships_transferred'] += cursor.rowcount

            # Update engine_components to point to canonical (avoid duplicates)
            self.db.conn.execute(
                """DELETE FROM engine_components
                   WHERE component_id = ? AND engine_id IN (
                       SELECT engine_id FROM engine_components WHERE component_id = ?
                   )""",
                (dup_id, canonical_id)
            )
            self.db.conn.execute(
                """UPDATE engine_components
                   SET component_id = ?
                   WHERE component_id = ?""",
                (canonical_id, dup_id)
            )

            # Delete the duplicate component
            self.db.delete_component(dup_id)
            result['deleted'] += 1

        self.db.conn.commit()
        return result

    def merge_all(self, threshold: float = DEFAULT_DEDUP_THRESHOLD, dry_run: bool = True) -> list[dict]:
        """
        Find and merge all duplicate groups.

        Args:
            threshold: Minimum similarity for duplicates
            dry_run: If True, report what would happen without modifying

        Returns:
            List of merge results
        """
        groups = self.find_duplicates(threshold)
        results = []

        for group in groups:
            result = self.merge_group(group, dry_run=dry_run)
            results.append(result)

        return results


def find_duplicate_engines(db: ArcFusionDB) -> list[tuple[Engine, Engine, str]]:
    """
    Find potentially duplicate engines.
    Returns list of (engine1, engine2, reason) tuples.
    """
    engines = db.list_engines()
    duplicates = []

    for i, e1 in enumerate(engines):
        norm1 = normalize_component_name(e1.name)
        for e2 in engines[i+1:]:
            norm2 = normalize_component_name(e2.name)

            # Check normalized name
            if norm1 == norm2:
                duplicates.append((e1, e2, "same normalized name"))
            elif norm1 in norm2 or norm2 in norm1:
                duplicates.append((e1, e2, "substring match"))

    return duplicates
