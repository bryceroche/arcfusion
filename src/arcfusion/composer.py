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
from .db import ArcFusionDB, Component, Recipe


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

# Compatibility scoring weights for component matching
INTERFACE_WEIGHT = 0.4      # Weight for interface shape compatibility
RELATIONSHIP_WEIGHT = 0.4   # Weight for learned relationship scores
CATEGORY_BONUS = 0.2        # Bonus for correct category ordering
CATEGORY_CLOSE_BONUS = 0.1  # Bonus for nearly correct ordering

# Configuration matching weights for dream scoring
CONFIG_MATCH_WEIGHT = 0.15  # Weight for matching known-good configurations
CONFIG_VALIDATED_BONUS = 0.05  # Extra bonus for validated configurations


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
        score += interface_score * INTERFACE_WEIGHT

        # Relationship from database (0-1)
        rel_cache = self._build_relationship_cache()
        rel_score = rel_cache.get(comp1.component_id, {}).get(comp2.component_id, 0.0)
        score += rel_score * RELATIONSHIP_WEIGHT

        # Category ordering bonus
        cat1 = get_component_category(comp1)
        cat2 = get_component_category(comp2)
        order1 = CATEGORY_ORDER.get(cat1, 4)
        order2 = CATEGORY_ORDER.get(cat2, 4)

        if order1 <= order2:  # Correct order
            score += CATEGORY_BONUS
        elif order1 - order2 <= 1:  # Close enough
            score += CATEGORY_CLOSE_BONUS

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
        comps1 = [c for cid in e1.component_ids if (c := self.db.get_component(cid))]
        comps2 = [c for cid in e2.component_ids if (c := self.db.get_component(cid))]

        all_parent_comps = comps1 + comps2
        if not all_parent_comps:
            return []

        # Group by category (deduplicate by component_id)
        by_category = {}
        seen_ids = set()
        for comp in all_parent_comps:
            if comp.component_id not in seen_ids:
                seen_ids.add(comp.component_id)
                cat = get_component_category(comp)
                by_category.setdefault(cat, []).append(comp)

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

    def _get_configuration_bonus(self, components: list[Component]) -> float:
        """
        Calculate bonus score based on matching known-good configurations.

        Configurations that have been extracted from working engines provide
        evidence that certain component combinations work well together.

        Returns:
            Bonus score (0.0 to CONFIG_MATCH_WEIGHT + CONFIG_VALIDATED_BONUS)
        """
        if len(components) < 2:
            return 0.0

        component_ids = [c.component_id for c in components]
        component_set = set(component_ids)

        # Find all configurations that overlap with our components
        all_configs = self.db.find_configurations()
        if not all_configs:
            return 0.0

        best_bonus = 0.0

        for config in all_configs:
            config_ids = set(config.component_ids)
            overlap = len(config_ids & component_set)

            if overlap < 2:
                continue

            # Check for contiguous sequence match (stronger signal)
            config_id_list = config.component_ids
            is_contiguous = False
            for start in range(len(component_ids) - len(config_id_list) + 1):
                if component_ids[start:start + len(config_id_list)] == config_id_list:
                    is_contiguous = True
                    break

            # Calculate match strength
            match_ratio = overlap / len(config_ids)

            if is_contiguous:
                # Full contiguous match - strong bonus
                bonus = CONFIG_MATCH_WEIGHT * config.config_score
            else:
                # Partial overlap - smaller bonus scaled by match ratio
                bonus = CONFIG_MATCH_WEIGHT * config.config_score * match_ratio * 0.5

            # Extra bonus for validated configurations
            if config.validated:
                bonus += CONFIG_VALIDATED_BONUS * match_ratio

            best_bonus = max(best_bonus, bonus)

        return best_bonus

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

        # Base score from component quality (guard against empty list)
        base_score = sum(c.usefulness_score for c in components) / max(len(components), 1)

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

        # Configuration bonus - prefer combinations that match known-good configs
        config_bonus = self._get_configuration_bonus(components)

        estimated_score = min(1.0, base_score + interface_bonus + diversity_bonus + pair_bonus + config_bonus)
        return components, estimated_score

    def create_recipe(
        self,
        name: str,
        strategy: str = "greedy",
        save_to_db: bool = True,
        **kwargs
    ) -> Recipe:
        """
        Dream up a new architecture and create a Recipe for the ML Agent.

        The Recipe contains:
        - Ordered list of component IDs
        - Assembly instructions (connections, residuals, shapes)
        - Strategy metadata for reproducibility

        Args:
            name: Name for the recipe
            strategy: Dream strategy to use
            save_to_db: Whether to store the recipe in the database
            **kwargs: Additional args passed to dream()

        Returns:
            Recipe object ready for ML Agent execution

        Raises:
            ValueError: If strategy is unknown or composition fails
        """
        components, estimated_score = self.dream(strategy, **kwargs)

        if not components:
            raise ValueError(f"Failed to compose architecture with strategy '{strategy}'")

        # Build assembly instructions
        assembly = self._build_assembly_instructions(components)

        # Track parent engines for crossover/mutate
        parent_engine_ids = []
        if strategy == "crossover":
            e1 = self.db.get_engine_by_name(kwargs.get("engine1_name", ""))
            e2 = self.db.get_engine_by_name(kwargs.get("engine2_name", ""))
            if e1:
                parent_engine_ids.append(e1.engine_id)
            if e2:
                parent_engine_ids.append(e2.engine_id)
        elif strategy == "mutate":
            e = self.db.get_engine_by_name(kwargs.get("engine_name", ""))
            if e:
                parent_engine_ids.append(e.engine_id)

        recipe = Recipe(
            name=name,
            component_ids=[c.component_id for c in components],
            assembly=assembly,
            strategy=strategy,
            estimated_score=estimated_score,
            parent_engine_ids=parent_engine_ids,
            notes=f"Dreamed with {strategy} strategy"
        )

        if save_to_db:
            self.db.add_recipe(recipe)

        return recipe

    def _build_assembly_instructions(self, components: list[Component]) -> dict:
        """
        Build assembly instructions for how components should be wired together.

        Returns:
            dict with:
            - connections: List of (from_id, to_id) tuples showing data flow
            - residuals: List of (from_id, to_id) tuples for skip connections
            - shapes: Dict mapping component_id to expected I/O shapes
            - categories: Dict mapping component_id to its category
            - notes: List of assembly hints for the ML Agent
        """
        if not components:
            return {}

        connections = []
        residuals = []
        shapes = {}
        categories = {}
        notes = []

        # Build sequential connections
        for i, comp in enumerate(components):
            categories[comp.component_id] = get_component_category(comp)

            # Record shapes
            shapes[comp.component_id] = {
                'in': comp.interface_in.get('shape', 'variable') if comp.interface_in else 'variable',
                'out': comp.interface_out.get('shape', 'variable') if comp.interface_out else 'variable'
            }

            # Connect to next component
            if i < len(components) - 1:
                next_comp = components[i + 1]
                connections.append((comp.component_id, next_comp.component_id))

        # Detect residual connection opportunities
        # Look for layer norm or attention followed by FFN (classic transformer pattern)
        for i, comp in enumerate(components):
            cat = categories[comp.component_id]

            # Residual around attention blocks
            if cat == 'attention' and i > 0:
                prev_comp = components[i - 1]
                # Add residual from before attention to after attention
                residuals.append((prev_comp.component_id, comp.component_id))
                notes.append(f"Residual connection around {comp.name}")

            # Residual around FFN blocks
            if cat == 'layer' and 'feed' in comp.name.lower():
                # Look back for a norm or attention to skip from
                for j in range(i - 1, max(0, i - 3), -1):
                    prev = components[j]
                    if categories[prev.component_id] in ('attention', 'layer'):
                        residuals.append((prev.component_id, comp.component_id))
                        notes.append(f"Residual connection around {comp.name}")
                        break

        # Add architecture-level notes
        cat_set = set(categories.values())
        if 'embedding' in cat_set and 'output' in cat_set:
            notes.append("Complete architecture with embedding and output layers")
        if 'attention' in cat_set:
            notes.append("Contains attention mechanism - consider causal masking for autoregressive")
        if 'efficiency' in cat_set:
            notes.append("Contains efficiency optimizations - may require special handling")

        return {
            'connections': connections,
            'residuals': residuals,
            'shapes': shapes,
            'categories': categories,
            'notes': notes
        }

    def recipe_to_components(self, recipe: Recipe) -> list[Component]:
        """
        Convert a Recipe back to a list of Component objects.

        Useful for code generation or validation.
        """
        components = []
        for cid in recipe.component_ids:
            comp = self.db.get_component(cid)
            if comp:
                components.append(comp)
        return components

    # -------------------------------------------------------------------------
    # Configuration extraction and management
    # -------------------------------------------------------------------------
    def extract_configurations_from_engine(
        self,
        engine_name: str,
        min_size: int = 2,
        max_size: int = None
    ) -> list['ComponentConfiguration']:
        """
        Extract common component configurations from an engine.

        Generates all contiguous sub-sequences of components that could be
        useful as reusable building blocks.

        Args:
            engine_name: Name of engine to extract from
            min_size: Minimum number of components in a config (default 2)
            max_size: Maximum number of components (default: engine size - 1)

        Returns:
            List of ComponentConfiguration objects
        """
        from .db import ComponentConfiguration

        engine = self.db.get_engine_by_name(engine_name)
        if not engine or not engine.component_ids:
            return []

        components = [c for cid in engine.component_ids if (c := self.db.get_component(cid))]

        if len(components) < min_size:
            return []

        if max_size is None:
            max_size = len(components) - 1

        configs = []

        # Generate all contiguous sub-sequences
        for size in range(min_size, min(max_size + 1, len(components))):
            for start in range(len(components) - size + 1):
                sub_components = components[start:start + size]
                comp_ids = [c.component_id for c in sub_components]
                comp_names = [c.name for c in sub_components]

                # Create a descriptive name
                if size <= 3:
                    name = " + ".join(comp_names)
                else:
                    name = f"{comp_names[0]} ... {comp_names[-1]} ({size} components)"

                # Calculate initial score based on component quality
                avg_score = sum(c.usefulness_score for c in sub_components) / len(sub_components)

                config = ComponentConfiguration(
                    name=name,
                    component_ids=comp_ids,
                    description=f"Extracted from {engine_name}",
                    source_engine_id=engine.engine_id,
                    config_score=avg_score,
                    validated=True  # From a known working engine
                )
                configs.append(config)

        return configs

    def save_configurations(self, configs: list['ComponentConfiguration']) -> int:
        """Save configurations to database, skipping duplicates."""
        saved = 0
        for config in configs:
            existing = self.db.get_configuration(config.config_id)
            if not existing:
                self.db.add_configuration(config)
                saved += 1
        return saved

    def find_matching_configurations(
        self,
        components: list['Component'],
        min_match_ratio: float = 0.5
    ) -> list[tuple['ComponentConfiguration', float]]:
        """
        Find configurations that match a subset of the given components.

        Args:
            components: List of components to match against
            min_match_ratio: Minimum ratio of config components that must match

        Returns:
            List of (configuration, match_ratio) tuples, sorted by match ratio
        """
        component_ids = {c.component_id for c in components}
        all_configs = self.db.find_configurations(validated=True)

        matches = []
        for config in all_configs:
            config_ids = set(config.component_ids)
            overlap = len(config_ids & component_ids)
            match_ratio = overlap / len(config_ids) if config_ids else 0

            if match_ratio >= min_match_ratio:
                matches.append((config, match_ratio))

        # Sort by match ratio descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
