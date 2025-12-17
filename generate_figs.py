"""
Helper script to regenerate the figures used in seminario.latex.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from utils import compute_aligned_umap_embedding, run_evolution
from visualizations.aligned_umap import plot as plot_aligned_umap
from visualizations.vector_field import plot as plot_vector_field


def save_fig(fig, name: str) -> None:
    fig.savefig(name, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}")


def main() -> None:
    plt.rcParams.update({"figure.dpi": 150})

    pop_size = 200
    n_generations = 50
    hidden_dim = 16
    mutation_rate = 0.05
    seed = 42
    lambda_align = 0.3

    print("Running evolution...")
    evo = run_evolution(pop_size, n_generations, hidden_dim, mutation_rate, seed)

    print("Computing aligned embeddings...")
    embedding_all, gen_labels, per_gen_embeddings = compute_aligned_umap_embedding(
        evo.weights_by_gen, lambda_align=lambda_align, random_state=seed
    )

    print("Aligned UMAP figure...")
    fig = plot_aligned_umap(
        evo.weights_by_gen,
        evo.fitness_by_gen,
        lambda_align=lambda_align,
        random_state=seed,
        fitness_bins=31,
    )
    save_fig(fig, "aligned_umap.png")

    print("Vector field figure...")
    fig = plot_vector_field(
        evo.weights_by_gen,
        evo.fitness_by_gen,
        lambda_align=lambda_align,
        grid_res=30,
        min_vectors_per_cell=1,
        random_state=seed,
        show_points=True,
        smoothing_sigma=1.0,
        subsample=1,
        vector_mode="stream",
        quantize_bins=None,
        fitness_bins=31,
        show_speed_colorbar=False,
        fill_empty_cells=True,
    )
    save_fig(fig, "vector_field.png")

    print("Fitness evolution figure...")
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    gens = np.arange(n_generations)
    ax.plot(gens, evo.mean_fitness, color="#3b65ff", linewidth=2.2, label="Fitness médio")
    ax.fill_between(
        gens,
        evo.mean_fitness - evo.std_fitness,
        evo.mean_fitness + evo.std_fitness,
        color="#3b65ff",
        alpha=0.18,
        label="±1 desvio",
    )
    ax.scatter(gens, evo.mean_fitness, color="#1f2a44", s=16)
    ax.set_xlabel("Geração")
    ax.set_ylabel("Fitness")
    ax.set_title("Evolução do fitness na neuroevolução")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(frameon=False)
    save_fig(fig, "fitness_evolution.png")

    print("Done.")


if __name__ == "__main__":
    main()
