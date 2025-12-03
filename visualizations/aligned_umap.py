"""
Aligned UMAP visualization as described by Cantareira et al. (2020).
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from utils import compute_aligned_umap_embedding


def plot(
    weights_by_gen: Sequence[np.ndarray],
    fitness_by_gen: Sequence[np.ndarray],
    lambda_align: float = 0.3,
    random_state: int = 42,
):
    """
    Plot aligned UMAP projections for all generations with two panels:
    - colored by generation
    - colored by fitness
    """
    embedding, gen_labels, _ = compute_aligned_umap_embedding(
        weights_by_gen, lambda_align=lambda_align, random_state=random_state
    )

    if len(fitness_by_gen) != len(weights_by_gen):
        raise ValueError("fitness_by_gen and weights_by_gen must have the same length.")

    fitness_concat = np.concatenate(fitness_by_gen) if fitness_by_gen else np.array([])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)

    # Left: colored by generation
    ax = axes[0]
    gens = np.unique(gen_labels)
    cmap_gens = plt.cm.get_cmap("tab20", max(len(gens), 1))
    for idx, g in enumerate(gens):
        pts = embedding[gen_labels == g]
        ax.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.75, color=cmap_gens(idx), label=f"G{g}")
    ax.set_title("Aligned UMAP by Generation")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    # Right: colored by fitness
    ax = axes[1]
    sc = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=fitness_concat,
        cmap="viridis",
        s=18,
        alpha=0.85,
    )
    ax.set_title("Aligned UMAP colored by Fitness")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Fitness")

    fig.tight_layout()
    return fig
