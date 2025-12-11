"""
Engine Composer - Generate new architecture configurations.

Strategies:
- Greedy: Start with best component, add compatible ones
- Random walk: Explore component space with temperature
- Crossover: Combine components from two engines
- Mutation: Swap components for alternatives
"""

import random
from .db import ArcFusionDB, Component


class EngineComposer:
    """Generate new engine configurations by combining compatible components."""

    def __init__(self, db: ArcFusionDB):
        self.db = db

    def greedy_compose(self, start_component: str = None, max_components: int = 6) -> list[Component]:
        """Build an engine greedily from the best components"""
        all_components = self.db.find_components()
        if not all_components:
            return []

        if start_component:
            matches = [c for c in all_components if start_component.lower() in c.name.lower()]
            current = matches[0] if matches else all_components[0]
        else:
            current = all_components[0]

        engine = [current]
        used_ids = {current.component_id}

        while len(engine) < max_components:
            compatible = self.db.get_compatible_components(current.component_id, min_score=0.7)
            candidates = [
                (self.db.get_component(cid), score)
                for cid, score in compatible
                if cid not in used_ids
            ]

            if not candidates:
                break

            best_comp, _ = max(candidates, key=lambda x: x[1])
            engine.append(best_comp)
            used_ids.add(best_comp.component_id)
            current = best_comp

        return engine

    def random_walk_compose(self, steps: int = 5, temperature: float = 1.0) -> list[Component]:
        """Random walk through component space, biased by compatibility"""
        all_components = self.db.find_components()
        if not all_components:
            return []

        weights = [c.usefulness_score ** temperature for c in all_components]
        current = random.choices(all_components, weights=weights)[0]

        engine = [current]
        used_ids = {current.component_id}

        for _ in range(steps - 1):
            compatible = self.db.get_compatible_components(current.component_id, min_score=0.5)
            candidates = [
                (self.db.get_component(cid), score)
                for cid, score in compatible
                if cid not in used_ids
            ]

            if not candidates:
                unused = [c for c in all_components if c.component_id not in used_ids]
                if not unused:
                    break
                current = random.choice(unused)
            else:
                weights = [score ** temperature for _, score in candidates]
                current = random.choices([c for c, _ in candidates], weights=weights)[0]

            engine.append(current)
            used_ids.add(current.component_id)

        return engine

    def crossover(self, engine1_name: str, engine2_name: str) -> list[Component]:
        """Create new engine by combining components from two parents"""
        e1 = self.db.get_engine_by_name(engine1_name)
        e2 = self.db.get_engine_by_name(engine2_name)

        if not e1 or not e2:
            return []

        all_comp_ids = list(set(e1.component_ids + e2.component_ids))
        split = random.randint(1, len(all_comp_ids) - 1)

        selected_ids = []
        for i, cid in enumerate(all_comp_ids):
            if (i < split and cid in e1.component_ids) or (i >= split and cid in e2.component_ids):
                selected_ids.append(cid)

        return [self.db.get_component(cid) for cid in selected_ids if self.db.get_component(cid)]

    def mutate(self, engine_name: str, mutation_rate: float = 0.2) -> list[Component]:
        """Mutate an engine by swapping components"""
        engine = self.db.get_engine_by_name(engine_name)
        if not engine:
            return []

        result = []
        for cid in engine.component_ids:
            comp = self.db.get_component(cid)
            if not comp:
                continue

            if random.random() < mutation_rate:
                compatible = self.db.get_compatible_components(cid, min_score=0.6)
                alternatives = [
                    self.db.get_component(alt_id)
                    for alt_id, _ in compatible
                    if alt_id != cid
                ]
                if alternatives:
                    comp = random.choice(alternatives)

            result.append(comp)

        return result

    def dream(self, strategy: str = "greedy", **kwargs) -> tuple[list[Component], float]:
        """
        Dream up a new engine configuration.
        Returns (components, estimated_score)
        """
        strategies = {
            "greedy": self.greedy_compose,
            "random": self.random_walk_compose,
            "mutate": self.mutate,
            "crossover": self.crossover,
        }

        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategies.keys())}")

        components = strategies[strategy](**kwargs)

        if not components:
            return [], 0.0

        base_score = sum(c.usefulness_score for c in components) / len(components)

        pair_bonus = 0.0
        for i, c1 in enumerate(components):
            for c2 in components[i + 1:]:
                compatible = self.db.get_compatible_components(c1.component_id, min_score=0.0)
                for cid, score in compatible:
                    if cid == c2.component_id:
                        pair_bonus += score * 0.1
                        break

        estimated_score = min(1.0, base_score + pair_bonus)
        return components, estimated_score
