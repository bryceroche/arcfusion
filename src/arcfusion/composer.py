"""
Engine Composer - Generate new architecture configurations.

Strategies:
- Greedy: Start with best component, add compatible ones
- Random walk: Explore component space with temperature
- Crossover: Combine components from two engines
- Mutation: Swap components for alternatives
"""

import random
import re
from .db import ArcFusionDB, Component


# Component categories and their typical ordering in an architecture
CATEGORY_ORDER = {
    'position': 0,      # Positional encodings first
    'embedding': 1,     # Embeddings
    'structure': 2,     # Encoder/decoder stacks
    'attention': 3,     # Attention mechanisms
    'layer': 4,         # Feed-forward, normalization
    'efficiency': 5,    # Optimizations like FlashAttention, KV-cache
    'output': 6,        # Output projections
    'training': 7,      # Training-specific (LR schedule, regularization)
}


def normalize_shape(shape_str: str) -> str:
    """Normalize a shape string for comparison."""
    if not shape_str:
        return ""
    # Remove spaces, lowercase
    s = shape_str.lower().replace(" ", "")
    # Normalize common patterns
    s = re.sub(r'\[batch,?', '[b,', s)
    s = re.sub(r'seq_?len', 'n', s)
    s = re.sub(r'tgt_?len', 'n', s)  # Target length same as seq
    s = re.sub(r'd_?model', 'd', s)
    s = re.sub(r'hidden', 'd', s)
    return s


def interfaces_compatible(comp1: Component, comp2: Component) -> tuple[bool, float]:
    """
    Check if comp1's output can connect to comp2's input.
    Returns (is_compatible, score) where score is 0-1.
    """
    out_shape = comp1.interface_out.get('shape', '') if comp1.interface_out else ''
    in_shape = comp2.interface_in.get('shape', '') if comp2.interface_in else ''

    # If either is missing, assume compatible but low score
    if not out_shape or not in_shape:
        return True, 0.3

    # Variable shapes are compatible with anything
    if 'variable' in out_shape.lower() or 'variable' in in_shape.lower():
        return True, 0.6

    # Scalar shapes have specific compatibility
    if 'scalar' in out_shape.lower() or 'scalar' in in_shape.lower():
        return out_shape.lower() == in_shape.lower(), 0.5

    # Normalize and compare
    norm_out = normalize_shape(out_shape)
    norm_in = normalize_shape(in_shape)

    if norm_out == norm_in:
        return True, 1.0

    # Check if shapes are compatible (same dimensionality, different labels)
    # e.g., [b,n,d] can connect to [b,n,d] regardless of naming
    out_dims = re.findall(r'\[([^\]]+)\]', norm_out)
    in_dims = re.findall(r'\[([^\]]+)\]', norm_in)

    if out_dims and in_dims:
        out_parts = out_dims[0].split(',')
        in_parts = in_dims[0].split(',')
        if len(out_parts) == len(in_parts):
            return True, 0.8

    return False, 0.0


def get_component_category(comp: Component) -> str:
    """Infer component category from name and description."""
    name_lower = comp.name.lower()
    desc_lower = (comp.description or '').lower()

    # Check for specific patterns
    if any(p in name_lower for p in ['position', 'encoding', 'embedding', 'rope', 'rotary', 'alibi']):
        if 'embed' in name_lower:
            return 'embedding'
        return 'position'
    if any(p in name_lower for p in ['attention', 'attn', 'mha', 'ssm', 'selective']):
        return 'attention'
    if any(p in name_lower for p in ['encoder', 'decoder', 'block', 'stack', 'layer ']):
        if 'norm' in name_lower:
            return 'layer'
        return 'structure'
    if any(p in name_lower for p in ['norm', 'feed-forward', 'ffn', 'mlp', 'gelu', 'relu', 'activation', 'residual', 'dropout']):
        return 'layer'
    if any(p in name_lower for p in ['flash', 'kv-cache', 'cache', 'efficient', 'sparse', 'quantiz']):
        return 'efficiency'
    if any(p in name_lower for p in ['output', 'projection', 'head', 'vocab', 'logit']):
        return 'output'
    if any(p in name_lower for p in ['learning rate', 'warmup', 'schedule', 'regulariz', 'label smooth', 'loss', 'optim']):
        return 'training'

    return 'layer'  # Default


class EngineComposer:
    """Generate new engine configurations by combining compatible components."""

    def __init__(self, db: ArcFusionDB):
        self.db = db
        self._relationship_cache = None

    def _build_relationship_cache(self) -> dict:
        """Build a cache of component relationships from the database."""
        if self._relationship_cache is not None:
            return self._relationship_cache

        cache = {}
        rows = self.db.conn.execute(
            "SELECT component1_id, component2_id, c2c_score FROM component_relationships"
        ).fetchall()

        for c1_id, c2_id, score in rows:
            if c1_id not in cache:
                cache[c1_id] = {}
            if c2_id not in cache:
                cache[c2_id] = {}
            cache[c1_id][c2_id] = score
            cache[c2_id][c1_id] = score  # Bidirectional

        self._relationship_cache = cache
        return cache

    def get_compatibility_score(self, comp1: Component, comp2: Component) -> float:
        """
        Get compatibility score between two components using:
        1. Interface compatibility (shapes match)
        2. Relationship score (from analysis)
        3. Category ordering (positional before attention, etc.)
        """
        score = 0.0

        # Interface compatibility (0-1)
        compatible, interface_score = interfaces_compatible(comp1, comp2)
        if not compatible:
            return 0.0
        score += interface_score * 0.4

        # Relationship from database (0-1)
        rel_cache = self._build_relationship_cache()
        rel_score = rel_cache.get(comp1.component_id, {}).get(comp2.component_id, 0.0)
        score += rel_score * 0.4

        # Category ordering bonus (0-0.2)
        cat1 = get_component_category(comp1)
        cat2 = get_component_category(comp2)
        order1 = CATEGORY_ORDER.get(cat1, 4)
        order2 = CATEGORY_ORDER.get(cat2, 4)

        if order1 <= order2:  # Correct order
            score += 0.2
        elif order1 - order2 <= 1:  # Close enough
            score += 0.1

        return min(score, 1.0)

    def get_interface_compatible_components(
        self,
        component_id: str,
        min_score: float = 0.5
    ) -> list[tuple[str, float]]:
        """
        Get components that are compatible with the given component.
        Uses interface matching and relationship data.
        """
        source = self.db.get_component(component_id)
        if not source:
            return []

        all_components = self.db.find_components()
        compatible = []

        for comp in all_components:
            if comp.component_id == component_id:
                continue

            score = self.get_compatibility_score(source, comp)
            if score >= min_score:
                compatible.append((comp.component_id, score))

        # Sort by score descending
        compatible.sort(key=lambda x: -x[1])
        return compatible

    def sort_by_architecture_order(self, components: list[Component]) -> list[Component]:
        """Sort components by their typical order in an architecture."""
        return sorted(components, key=lambda c: CATEGORY_ORDER.get(get_component_category(c), 4))

    def greedy_compose(self, start_component: str = None, max_components: int = 6) -> list[Component]:
        """Build an engine greedily from the best components using interface matching."""
        all_components = self.db.find_components()
        if not all_components:
            return []

        if start_component:
            matches = [c for c in all_components if start_component.lower() in c.name.lower()]
            current = matches[0] if matches else all_components[0]
        else:
            # Start with a positional encoding or embedding component
            positional = [c for c in all_components if get_component_category(c) in ('position', 'embedding')]
            current = positional[0] if positional else all_components[0]

        engine = [current]
        used_ids = {current.component_id}
        used_categories = {get_component_category(current)}

        while len(engine) < max_components:
            # Use interface-aware compatibility
            compatible = self.get_interface_compatible_components(current.component_id, min_score=0.4)
            candidates = [
                (self.db.get_component(cid), score)
                for cid, score in compatible
                if cid not in used_ids
            ]

            if not candidates:
                # Try to find any unused component that fits
                unused = [c for c in all_components if c.component_id not in used_ids]
                # Prefer components from categories not yet used
                diverse = [c for c in unused if get_component_category(c) not in used_categories]
                if diverse:
                    # Pick the highest scoring one
                    current = max(diverse, key=lambda c: c.usefulness_score)
                elif unused:
                    current = max(unused, key=lambda c: c.usefulness_score)
                else:
                    break
            else:
                # Prefer diversity - boost components from new categories
                for i, (comp, score) in enumerate(candidates):
                    if get_component_category(comp) not in used_categories:
                        candidates[i] = (comp, score + 0.1)

                best_comp, _ = max(candidates, key=lambda x: x[1])
                current = best_comp

            engine.append(current)
            used_ids.add(current.component_id)
            used_categories.add(get_component_category(current))

        # Sort by architecture order before returning
        return self.sort_by_architecture_order(engine)

    def random_walk_compose(self, steps: int = 5, temperature: float = 1.0) -> list[Component]:
        """Random walk through component space, biased by interface compatibility."""
        if temperature < 0:
            raise ValueError("temperature must be >= 0")

        all_components = self.db.find_components()
        if not all_components:
            return []

        # Start preferentially with position/embedding components
        start_candidates = [c for c in all_components if get_component_category(c) in ('position', 'embedding')]
        if not start_candidates:
            start_candidates = all_components

        # Ensure weights are valid (non-zero) for random.choices
        weights = [max(0.01, c.usefulness_score) ** temperature for c in start_candidates]
        current = random.choices(start_candidates, weights=weights)[0]

        engine = [current]
        used_ids = {current.component_id}
        used_categories = {get_component_category(current)}

        for _ in range(steps - 1):
            # Use interface-aware compatibility
            compatible = self.get_interface_compatible_components(current.component_id, min_score=0.3)
            candidates = [
                (self.db.get_component(cid), score)
                for cid, score in compatible
                if cid not in used_ids
            ]

            if not candidates:
                unused = [c for c in all_components if c.component_id not in used_ids]
                if not unused:
                    break
                # Prefer diverse categories
                diverse = [c for c in unused if get_component_category(c) not in used_categories]
                pool = diverse if diverse else unused
                current = random.choice(pool)
            else:
                # Boost diversity
                boosted = []
                for comp, score in candidates:
                    if get_component_category(comp) not in used_categories:
                        boosted.append((comp, score + 0.15))
                    else:
                        boosted.append((comp, score))

                # Ensure weights are valid (non-zero) for random.choices
                weights = [max(0.01, score) ** temperature for _, score in boosted]
                current = random.choices([c for c, _ in boosted], weights=weights)[0]

            engine.append(current)
            used_ids.add(current.component_id)
            used_categories.add(get_component_category(current))

        # Sort by architecture order
        return self.sort_by_architecture_order(engine)

    def crossover(self, engine1_name: str, engine2_name: str) -> list[Component]:
        """
        Create new engine by combining components from two parents with interface awareness.

        Raises:
            ValueError: If either engine is not found in the database.
        """
        e1 = self.db.get_engine_by_name(engine1_name)
        e2 = self.db.get_engine_by_name(engine2_name)

        if not e1:
            raise ValueError(f"Engine not found: '{engine1_name}'")
        if not e2:
            raise ValueError(f"Engine not found: '{engine2_name}'")

        # Get all components from both engines
        comps1 = [self.db.get_component(cid) for cid in e1.component_ids if self.db.get_component(cid)]
        comps2 = [self.db.get_component(cid) for cid in e2.component_ids if self.db.get_component(cid)]

        all_parent_comps = comps1 + comps2
        if not all_parent_comps:
            return []

        # Group by category
        by_category = {}
        for comp in all_parent_comps:
            cat = get_component_category(comp)
            if cat not in by_category:
                by_category[cat] = []
            if comp not in by_category[cat]:
                by_category[cat].append(comp)

        # For each category that exists, randomly pick from available components
        selected = []
        for cat in CATEGORY_ORDER.keys():
            if cat in by_category and by_category[cat]:
                # Take 1 component per category for cleaner architectures
                selected.append(random.choice(by_category[cat]))

        # If we got nothing from category iteration, take top components from both parents
        if not selected:
            # Fallback: take highest-scoring components from each parent
            sorted_comps = sorted(all_parent_comps, key=lambda c: c.usefulness_score, reverse=True)
            selected = sorted_comps[:min(6, len(sorted_comps))]

        # Sort by architecture order
        selected = self.sort_by_architecture_order(selected)

        # Try to filter by interface compatibility, but keep originals if too aggressive
        if len(selected) > 1:
            final = [selected[0]]
            for i in range(1, len(selected)):
                compatible, score = interfaces_compatible(final[-1], selected[i])
                # Accept if compatible OR if score is reasonable (lenient)
                if compatible or score >= 0.3:
                    final.append(selected[i])

            # Only use filtered result if it kept at least half the components
            if len(final) >= len(selected) // 2 and len(final) >= 2:
                selected = final

        return selected

    def mutate(self, engine_name: str, mutation_rate: float = 0.2) -> list[Component]:
        """
        Mutate an engine by swapping components with interface-compatible alternatives.

        Raises:
            ValueError: If engine is not found in the database.
        """
        engine = self.db.get_engine_by_name(engine_name)
        if not engine:
            raise ValueError(f"Engine not found: '{engine_name}'")

        result = []
        for cid in engine.component_ids:
            comp = self.db.get_component(cid)
            if not comp:
                continue

            if random.random() < mutation_rate:
                # Find alternatives in the same category with compatible interfaces
                category = get_component_category(comp)
                all_components = self.db.find_components()
                alternatives = [
                    c for c in all_components
                    if c.component_id != cid
                    and get_component_category(c) == category
                ]

                # Filter by interface compatibility
                if result:
                    # Must be compatible with previous component
                    alternatives = [
                        c for c in alternatives
                        if interfaces_compatible(result[-1], c)[0]
                    ]

                if alternatives:
                    # Prefer higher scoring alternatives
                    weights = [c.usefulness_score for c in alternatives]
                    comp = random.choices(alternatives, weights=weights)[0]

            result.append(comp)

        return self.sort_by_architecture_order(result)

    def dream(self, strategy: str = "greedy", **kwargs) -> tuple[list[Component], float]:
        """
        Dream up a new engine configuration.

        Returns:
            (components, estimated_score) - If components is empty, composition failed.

        Raises:
            ValueError: If strategy is unknown or required kwargs are missing.
        """
        strategies = {
            "greedy": self.greedy_compose,
            "random": self.random_walk_compose,
            "mutate": self.mutate,
            "crossover": self.crossover,
        }

        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategies.keys())}")

        # Validate required kwargs for each strategy
        if strategy == "crossover":
            if not kwargs.get("engine1_name") or not kwargs.get("engine2_name"):
                raise ValueError("crossover strategy requires engine1_name and engine2_name")
        if strategy == "mutate":
            if not kwargs.get("engine_name"):
                raise ValueError("mutate strategy requires engine_name")

        components = strategies[strategy](**kwargs)

        if not components:
            # Return empty with score of -1 to indicate failure (vs 0.0 for very low score)
            return [], -1.0

        # Base score from component quality
        base_score = sum(c.usefulness_score for c in components) / len(components)

        # Interface flow bonus - components should connect well in sequence
        interface_bonus = 0.0
        for i in range(len(components) - 1):
            compatible, score = interfaces_compatible(components[i], components[i + 1])
            if compatible:
                interface_bonus += score * 0.05

        # Category diversity bonus - good architectures have multiple component types
        categories = set(get_component_category(c) for c in components)
        diversity_bonus = len(categories) * 0.02

        # Relationship bonus - components that have worked together before
        pair_bonus = 0.0
        rel_cache = self._build_relationship_cache()
        for i, c1 in enumerate(components):
            for c2 in components[i + 1:]:
                rel_score = rel_cache.get(c1.component_id, {}).get(c2.component_id, 0.0)
                if rel_score > 0:
                    pair_bonus += rel_score * 0.03

        estimated_score = min(1.0, base_score + interface_bonus + diversity_bonus + pair_bonus)
        return components, estimated_score
