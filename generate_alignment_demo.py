"""
Generate a side-by-side figure illustrating aligned vs non-aligned UMAP projections.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from utils import compute_aligned_umap_embedding, run_evolution


def compute_unaligned_umap(weights_by_gen, random_state: int = 42, n_neighbors: int = 15, min_dist: float = 0.1):
    """
    Compute per-generation UMAP projections without temporal alignment (lambda=0).
    """
    return compute_aligned_umap_embedding(
        weights_by_gen,
        lambda_align=0.0,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )


def main():
    plt.rcParams.update({"figure.dpi": 150})

    pop_size = 200
    n_generations = 50
    hidden_dim = 16
    mutation_rate = 0.05
    seed = 42

    evo = run_evolution(pop_size, n_generations, hidden_dim, mutation_rate, seed)

    print("Computing unaligned UMAP...")
    emb_unaligned, gen_unaligned, _ = compute_unaligned_umap(evo.weights_by_gen, random_state=seed)

    print("Computing aligned UMAP...")
    emb_aligned, gen_aligned, _ = compute_aligned_umap_embedding(
        evo.weights_by_gen, lambda_align=0.3, random_state=seed
    )

    cmap = plt.cm.turbo
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)

    for ax, emb, gens, title in [
        (axes[0], emb_unaligned, gen_unaligned, "UMAP sem alinhamento"),
        (axes[1], emb_aligned, gen_aligned, "UMAP com alinhamento temporal (lambda=0.3)"),
    ]:
        norm = plt.Normalize(vmin=gens.min() if len(gens) else 0, vmax=gens.max() if len(gens) else 1)
        ax.scatter(
            emb[:, 0],
            emb[:, 1],
            c=gens,
            cmap=cmap,
            norm=norm,
            s=10,
            alpha=0.6,
            edgecolors="none",
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_generations - 1)),
        ax=axes,
        orientation="horizontal",
        fraction=0.08,
        pad=0.1,
    )
    cbar.set_label("Geração")

    fig.suptitle("Aligned vs. Non-Aligned UMAP Projections", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig("aligned_vs_unaligned.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved aligned_vs_unaligned.png")


if __name__ == "__main__":
    main()
