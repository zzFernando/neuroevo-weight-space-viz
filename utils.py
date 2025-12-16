"""
Utility classes and helpers for running the neuroevolution experiment and
preparing aligned embeddings used by the visualizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import warnings

import numpy as np
import umap
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# Silencia aviso do UMAP sobre n_jobs ser forçado para 1 quando random_state é definido.
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state")


@dataclass
class EvolutionResult:
    """Container for neuroevolution data collected per generation."""

    weights_by_gen: List[np.ndarray]
    fitness_by_gen: List[np.ndarray]
    best_indices: np.ndarray
    mean_fitness: np.ndarray
    std_fitness: np.ndarray


class NeuroEvoMoons:
    """
    Simple neuroevolution of a one-hidden-layer MLP for make_moons.
    """

    def __init__(
        self,
        pop_size: int = 80,
        input_dim: int = 2,
        hidden_dim: int = 16,
        output_dim: int = 1,
        mutation_rate: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate

        self.shapes = [
            (input_dim, hidden_dim),
            (hidden_dim, output_dim),
        ]

        X, y = make_moons(n_samples=400, noise=0.25, random_state=seed)
        self.scaler = StandardScaler().fit(X)
        self.X = self.scaler.transform(X)
        self.y = y.reshape(-1, 1)

        self.population = [self.random_individual() for _ in range(pop_size)]

    # ---- Representation helpers -------------------------------------------------
    def random_individual(self) -> List[np.ndarray]:
        return [self.rng.normal(0, 0.5, size=s) for s in self.shapes]

    def flatten(self, individual: Sequence[np.ndarray]) -> np.ndarray:
        return np.concatenate([w.ravel() for w in individual])

    # ---- Forward / Fitness ------------------------------------------------------
    def forward(self, individual: Sequence[np.ndarray], X: np.ndarray) -> np.ndarray:
        w1, w2 = individual
        h = np.tanh(X @ w1)
        o = h @ w2
        return 1.0 / (1.0 + np.exp(-o))  # sigmoid

    def evaluate(self, individual: Sequence[np.ndarray]) -> float:
        preds = self.forward(individual, self.X)
        eps = 1e-8
        loss = -(self.y * np.log(preds + eps) + (1 - self.y) * np.log(1 - preds + eps)).mean()
        return -float(loss)

    # ---- Mutation / Evolution ---------------------------------------------------
    def mutate(self, individual: Sequence[np.ndarray]) -> List[np.ndarray]:
        return [w + self.rng.normal(0, self.mutation_rate, size=w.shape) for w in individual]

    def evolve_one_generation(self, elite_frac: float = 0.2) -> Tuple[np.ndarray, float]:
        fitness = np.array([self.evaluate(ind) for ind in self.population])

        n_elite = max(2, int(self.pop_size * elite_frac))
        elite_idx = np.argsort(fitness)[-n_elite:]
        elites = [self.population[i] for i in elite_idx]

        new_pop: List[List[np.ndarray]] = elites.copy()
        while len(new_pop) < self.pop_size:
            parent = elites[self.rng.integers(0, len(elites))]
            new_pop.append(self.mutate(parent))

        self.population = new_pop

        new_fitness = np.array([self.evaluate(ind) for ind in self.population])
        best_idx = int(np.argmax(new_fitness))
        best_fit = float(new_fitness[best_idx])
        return best_idx, best_fit


def run_evolution(
    pop_size: int,
    n_generations: int,
    hidden_dim: int,
    mutation_rate: float,
    seed: int,
) -> EvolutionResult:
    """
    Run neuroevolution and collect flattened weights/fitness per generation.
    """
    ne = NeuroEvoMoons(
        pop_size=pop_size,
        hidden_dim=hidden_dim,
        mutation_rate=mutation_rate,
        seed=seed,
    )

    weights_by_gen: List[np.ndarray] = []
    fitness_by_gen: List[np.ndarray] = []
    best_indices: List[int] = []
    mean_fitness: List[float] = []
    std_fitness: List[float] = []

    # Generation 0
    fitness0 = np.array([ne.evaluate(ind) for ind in ne.population])
    weights_by_gen.append(np.stack([ne.flatten(ind) for ind in ne.population]))
    fitness_by_gen.append(fitness0)
    best_indices.append(int(np.argmax(fitness0)))
    mean_fitness.append(float(fitness0.mean()))
    std_fitness.append(float(fitness0.std()))

    # Subsequent generations
    for _ in range(1, n_generations):
        best_idx, _ = ne.evolve_one_generation()
        fitness = np.array([ne.evaluate(ind) for ind in ne.population])

        weights_by_gen.append(np.stack([ne.flatten(ind) for ind in ne.population]))
        fitness_by_gen.append(fitness)
        best_indices.append(int(best_idx))
        mean_fitness.append(float(fitness.mean()))
        std_fitness.append(float(fitness.std()))

    return EvolutionResult(
        weights_by_gen=weights_by_gen,
        fitness_by_gen=fitness_by_gen,
        best_indices=np.array(best_indices, dtype=int),
        mean_fitness=np.array(mean_fitness, dtype=float),
        std_fitness=np.array(std_fitness, dtype=float),
    )


def compute_aligned_umap_embedding(
    weights_by_gen: Sequence[np.ndarray],
    lambda_align: float = 0.3,
    random_state: int = 42,
    n_neighbors: int = 30,
    min_dist: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Compute per-generation UMAP projections and apply a simple temporal alignment.

    Returns the concatenated embedding, generation labels per point, and the list
    of aligned embeddings per generation (used by other visualizations).
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="euclidean",
        random_state=random_state,
    )

    aligned_embeddings: List[np.ndarray] = []
    gen_labels: List[np.ndarray] = []
    stacked: List[np.ndarray] = []

    proj_ref: np.ndarray | None = None

    for g_idx, weights in enumerate(weights_by_gen):
        proj = reducer.fit_transform(weights)

        if proj_ref is None:
            proj_aligned = proj
            proj_ref = proj
        else:
            proj_aligned = proj - lambda_align * (proj - proj_ref)
            proj_ref = (proj_ref * g_idx + proj_aligned) / (g_idx + 1)

        aligned_embeddings.append(proj_aligned)
        gen_labels.append(np.full(len(proj_aligned), g_idx, dtype=int))
        stacked.append(proj_aligned)

    if stacked:
        embedding_all = np.vstack(stacked)
        gen_labels_all = np.concatenate(gen_labels)
    else:
        embedding_all = np.empty((0, 2))
        gen_labels_all = np.empty((0,), dtype=int)

    return embedding_all, gen_labels_all, aligned_embeddings
