from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import umap
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state")


@dataclass
class EvolutionResult:
    weights_by_gen: List[np.ndarray]
    fitness_by_gen: List[np.ndarray]
    best_indices: np.ndarray
    mean_fitness: np.ndarray
    std_fitness: np.ndarray


def run_evolution_benchmark(
    benchmark_name: str,
    pop_size: int,
    n_generations: int,
    hidden_dim: int,
    mutation_rate: float,
    seed: int,
    elite_frac: float = 0.2,
    min_elite: int = 2,
) -> EvolutionResult:
    from benchmarks import REGISTRY

    cls = REGISTRY[benchmark_name]
    bench = cls(
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

    fitness0 = np.array([bench.evaluate(ind) for ind in bench.population])
    weights_by_gen.append(np.stack([bench.flatten(ind) for ind in bench.population]))
    fitness_by_gen.append(fitness0)
    best_indices.append(int(np.argmax(fitness0)))
    mean_fitness.append(float(fitness0.mean()))
    std_fitness.append(float(fitness0.std()))

    for _ in range(1, n_generations):
        best_idx, _ = bench.evolve_one_generation(elite_frac=elite_frac, min_elite=min_elite)
        fitness = np.array([bench.evaluate(ind) for ind in bench.population])
        weights_by_gen.append(np.stack([bench.flatten(ind) for ind in bench.population]))
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
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    pca_dims: Optional[int] = None,
    metric: str = "euclidean",
    metric_kwds: Optional[dict] = None,
    n_components: int = 2,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Joint UMAP embedding with reference-anchored temporal alignment.

    Fits UMAP once on all generations stacked together, then projects each
    generation independently and applies:
        Z_aligned_k = (1 - λ) * Z_k + λ * Z_ref
    where Z_ref is a running mean of previously aligned frames.

    λ=0 → pure joint UMAP (no temporal adjustment).
    λ=0.8 → recommended setting (paper §3).
    λ=1 → fully collapses to the running reference.

    Returns (embedding_all, gen_labels, per_gen_embeddings).
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        metric_kwds=metric_kwds,
        random_state=random_state,
    )

    aligned_embeddings: List[np.ndarray] = []
    gen_labels: List[np.ndarray] = []
    stacked: List[np.ndarray] = []
    proj_ref: np.ndarray | None = None

    pca: Optional[PCA] = None
    if pca_dims is not None:
        all_weights = np.vstack(list(weights_by_gen))
        actual_dims = min(pca_dims, all_weights.shape[1], all_weights.shape[0] - 1)
        pca = PCA(n_components=actual_dims, random_state=random_state)
        pca.fit(all_weights)

    for g_idx, weights in enumerate(weights_by_gen):
        w = pca.transform(weights) if pca is not None else weights
        proj = reducer.fit_transform(w)

        if proj_ref is None:
            proj_aligned = proj
            proj_ref = proj.copy()
        else:
            proj_aligned = proj - lambda_align * (proj - proj_ref)
            proj_ref = (proj_ref * g_idx + proj_aligned) / (g_idx + 1)

        aligned_embeddings.append(proj_aligned)
        gen_labels.append(np.full(len(proj_aligned), g_idx, dtype=int))
        stacked.append(proj_aligned)

    embedding_all = np.vstack(stacked) if stacked else np.empty((0, 2))
    gen_labels_all = np.concatenate(gen_labels) if gen_labels else np.empty((0,), dtype=int)

    return embedding_all, gen_labels_all, aligned_embeddings
