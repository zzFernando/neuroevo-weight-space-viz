"""
Aligned UMAP visualization as described by Cantareira et al. (2020).
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from utils import compute_aligned_umap_embedding
from .common import AdaptiveColorNorm, build_fitness_quantile_colormap, scatter_2d


def plot(
    weights_by_gen: Sequence[np.ndarray],
    fitness_by_gen: Sequence[np.ndarray],
    lambda_align: float = 0.3,
    random_state: int = 42,
    cmap_gen=plt.cm.plasma,
    cmap_fit=plt.cm.cividis,
    norm_mode: str = "power",
    gamma: float = 0.3,
    fitness_bins: int = 9,
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
    cmap_fit, fit_norm, boundaries = build_fitness_quantile_colormap(
        fitness_concat, n_bins=fitness_bins, cmap_name=cmap_fit.name if hasattr(cmap_fit, "name") else "plasma"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    # Left: colored by generation (temporal gradient)
    ax = axes[0]
    norm_gen = plt.Normalize(vmin=gen_labels.min(), vmax=gen_labels.max() if len(gen_labels) else 1)
    sc_gen = scatter_2d(ax, embedding, gen_labels, cmap_gen, norm_gen, size=10, base_alpha=0.5)
    ax.set_title("Aligned UMAP — Generation", fontsize=13)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    cbar_gen = fig.colorbar(sc_gen, ax=ax, orientation="horizontal", pad=0.12, fraction=0.05)
    cbar_gen.set_label("Generation index")

    # Right: colored by fitness
    ax = axes[1]
    sc = scatter_2d(ax, embedding, fitness_concat, cmap_fit, fit_norm, size=10, base_alpha=0.65, density_alpha=True)
    ax.set_title("Aligned UMAP — Fitness", fontsize=13)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.12, fraction=0.05)
    cbar.set_label("Fitness (quantile bins)")
    tick_positions = np.linspace(0, len(boundaries) - 2, 5, dtype=int)
    tick_values = [0.5 * (boundaries[i] + boundaries[i + 1]) for i in tick_positions]
    cbar.set_ticks(tick_values)
    cbar.ax.set_xticklabels([f"{v:.2f}" for v in tick_values])

    fig.tight_layout()
    return fig
