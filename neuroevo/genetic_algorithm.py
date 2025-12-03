"""Minimal genetic algorithm focused on tracking weight-space dynamics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np

from .mlp import FixedMLP


@dataclass
class GAHistory:
    generations: List[int]
    best_fitness: List[float]
    mean_fitness: List[float]
    diversity: List[float]


class GeneticAlgorithm:
    """Genetic algorithm that keeps the population history for visualization."""

    def __init__(self,
                 layer_sizes: List[int],
                 population_size: int = 80,
                 mutation_rate: float = 0.1,
                 mutation_std: float = 0.4,
                 elite_size: int = 5,
                 tournament_size: int = 4,
                 random_seed: Optional[int] = None):
        self.layer_sizes = layer_sizes
        self.population_size = population_size
        self.mutation_rate = float(mutation_rate)
        self.mutation_std = float(mutation_std)
        self.elite_size = int(elite_size)
        self.tournament_size = int(tournament_size)
        self.rng = np.random.default_rng(random_seed)

        template = FixedMLP(layer_sizes)
        self.genome_size = template.total_weights

        self.population = self.rng.normal(
            loc=0.0, scale=0.5, size=(population_size, self.genome_size)
        )
        self.fitness_scores = np.zeros(population_size, dtype=float)
        self.generation = 0

        self.best_individual_history: List[np.ndarray] = []
        self.population_history: List[np.ndarray] = []
        self.fitness_history: List[np.ndarray] = []
        self.history = GAHistory([], [], [], [])

    def evaluate_fitness(self, X: np.ndarray, y: np.ndarray) -> None:
        mlp = FixedMLP(self.layer_sizes)
        for i in range(self.population_size):
            mlp.set_weights(self.population[i])
            preds = mlp.predict(X)
            self.fitness_scores[i] = float(np.mean(preds == y))

    def _tournament_selection(self) -> np.ndarray:
        k = min(self.tournament_size, self.population_size)
        indices = self.rng.choice(self.population_size, size=k, replace=False)
        winner = indices[np.argmax(self.fitness_scores[indices])]
        return self.population[winner].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        mask = self.rng.random(self.genome_size) < 0.5
        return np.where(mask, parent1, parent2)

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        mask = self.rng.random(self.genome_size) < self.mutation_rate
        noise = self.rng.normal(scale=self.mutation_std, size=self.genome_size)
        individual[mask] += noise[mask]
        return individual

    def _population_diversity(self) -> float:
        if self.population.shape[0] < 2:
            return 0.0
        diffs = self.population[:, None, :] - self.population[None, :, :]
        dists = np.sqrt(np.sum(diffs * diffs, axis=2))
        iu = np.triu_indices_from(dists, k=1)
        return float(np.mean(dists[iu]))

    def _record_history(self) -> None:
        best_idx = int(np.argmax(self.fitness_scores))
        best = self.population[best_idx].copy()
        self.best_individual_history.append(best)
        self.population_history.append(self.population.copy())
        self.fitness_history.append(self.fitness_scores.copy())

        self.history.generations.append(self.generation)
        self.history.best_fitness.append(float(self.fitness_scores[best_idx]))
        self.history.mean_fitness.append(float(np.mean(self.fitness_scores)))
        self.history.diversity.append(self._population_diversity())

    def evolve_one_generation(self) -> None:
        sorted_idx = np.argsort(self.fitness_scores)[::-1]
        elites = self.population[sorted_idx[:self.elite_size]].copy()
        offspring = [elite for elite in elites]

        while len(offspring) < self.population_size:
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            child = self._mutate(self._crossover(parent1, parent2))
            offspring.append(child)

        self.population = np.vstack(offspring)[:self.population_size]
        self.generation += 1

    def run(self,
            X: np.ndarray,
            y: np.ndarray,
            generations: int = 150,
            verbose: bool = True) -> GAHistory:
        self.evaluate_fitness(X, y)
        self._record_history()
        if verbose:
            print(f"Gen {self.generation:03d} | "
                  f"Best={self.history.best_fitness[-1]:.3f} "
                  f"Mean={self.history.mean_fitness[-1]:.3f} "
                  f"Diversity={self.history.diversity[-1]:.3f}")

        for _ in range(generations):
            self.evolve_one_generation()
            self.evaluate_fitness(X, y)
            self._record_history()
            if verbose:
                print(f"Gen {self.generation:03d} | "
                      f"Best={self.history.best_fitness[-1]:.3f} "
                      f"Mean={self.history.mean_fitness[-1]:.3f} "
                      f"Diversity={self.history.diversity[-1]:.3f}")

        return self.history

    def get_best_individual(self) -> Dict[str, np.ndarray | float]:
        idx = int(np.argmax(self.fitness_scores))
        return {
            "weights": self.population[idx].copy(),
            "fitness": float(self.fitness_scores[idx])
        }
