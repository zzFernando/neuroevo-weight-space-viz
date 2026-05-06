from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


class NeuroEvoBase:
    """
    Shared neuroevolution mechanics. Subclasses must set:
        self.shapes, self.rng, self.pop_size, self.mutation_rate,
        self.weight_init_mean, self.weight_init_std, self.population
    and implement evaluate().
    """

    shapes: list
    rng: np.random.Generator
    pop_size: int
    mutation_rate: float
    weight_init_mean: float = 0.0
    weight_init_std: float = 0.5
    population: list

    def random_individual(self) -> List[np.ndarray]:
        return [
            self.rng.normal(self.weight_init_mean, self.weight_init_std, size=s)
            for s in self.shapes
        ]

    def flatten(self, individual: Sequence[np.ndarray]) -> np.ndarray:
        return np.concatenate([w.ravel() for w in individual])

    def mutate(self, individual: Sequence[np.ndarray]) -> List[np.ndarray]:
        return [w + self.rng.normal(0, self.mutation_rate, size=w.shape) for w in individual]

    def evaluate(self, individual: Sequence[np.ndarray]) -> float:
        raise NotImplementedError

    def evolve_one_generation(
        self, elite_frac: float = 0.2, min_elite: int = 2
    ) -> Tuple[int, float]:
        fitness = np.array([self.evaluate(ind) for ind in self.population])
        n_elite = max(min_elite, int(self.pop_size * elite_frac))
        elite_idx = np.argsort(fitness)[-n_elite:]
        elites = [self.population[i] for i in elite_idx]

        new_pop: list = elites.copy()
        while len(new_pop) < self.pop_size:
            parent = elites[self.rng.integers(0, len(elites))]
            new_pop.append(self.mutate(parent))
        self.population = new_pop

        new_fitness = np.array([self.evaluate(ind) for ind in self.population])
        best_idx = int(np.argmax(new_fitness))
        return best_idx, float(new_fitness[best_idx])
