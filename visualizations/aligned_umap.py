"""Aligned UMAP visualization used in the tool."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import compute_aligned_umap_embedding
from .common import (
    AdaptiveColorNorm,
    build_fitness_quantile_colormap,
    get_discrete_cmap,
    mpl_cmap_to_plotly_scale,
    scatter_2d,
)


def plot(
    weights_by_gen: Sequence[np.ndarray],
    fitness_by_gen: Sequence[np.ndarray],
    lambda_align: float = 0.3,
    random_state: int = 42,
    cmap_gen=plt.cm.turbo,
    cmap_fit=None,
    norm_mode: str = "power",
    gamma: float = 0.3,
    fitness_bins: int = 31,
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
    if cmap_fit is None:
        cmap_fit = get_discrete_cmap("fitness_map", n=24)
    cmap_fit, fit_norm, boundaries = build_fitness_quantile_colormap(
        fitness_concat, n_bins=fitness_bins, cmap_name_or_obj=cmap_fit
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
    cbar.set_label("Fitness")
    tick_positions = np.linspace(0, len(boundaries) - 2, 5, dtype=int)
    tick_values = [0.5 * (boundaries[i] + boundaries[i + 1]) for i in tick_positions]
    cbar.set_ticks(tick_values)
    cbar.ax.set_xticklabels([f"{v:.2f}" for v in tick_values])

    fig.tight_layout()
    return fig


def plot_interactive(
    weights_by_gen: Sequence[np.ndarray],
    fitness_by_gen: Sequence[np.ndarray],
    lambda_align: float = 0.3,
    random_state: int = 42,
    cmap_gen=plt.cm.turbo,
    cmap_fit=None,
    norm_mode: str = "power",
    gamma: float = 0.3,
    fitness_bins: int = 31,
):
    """
    Interactive Plotly version of the aligned UMAP scatter plots.
    """
    embedding, gen_labels, _ = compute_aligned_umap_embedding(
        weights_by_gen, lambda_align=lambda_align, random_state=random_state
    )
    fitness_concat = np.concatenate(fitness_by_gen) if fitness_by_gen else np.array([])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Aligned UMAP — Generation", "Aligned UMAP — Fitness"],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )

    if len(embedding):
        colorscale_gen = mpl_cmap_to_plotly_scale(cmap_gen)
        gen_min, gen_max = gen_labels.min(), gen_labels.max()
        customdata = (
            np.stack([gen_labels, fitness_concat], axis=1) if len(fitness_concat) == len(gen_labels) else None
        )

        fig.add_trace(
            go.Scattergl(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode="markers",
                marker=dict(
                    size=7,
                    color=gen_labels,
                    colorscale=colorscale_gen,
                    colorbar=dict(title="Generation", x=0.45),
                    cmin=gen_min,
                    cmax=gen_max,
                    opacity=0.75,
                ),
                hovertemplate="Gen %{customdata[0]}<br>Fitness %{customdata[1]:.3f}<extra></extra>"
                if customdata is not None
                else "Gen %{marker.color}<extra></extra>",
                customdata=customdata,
            ),
            row=1,
            col=1,
        )

        if cmap_fit is None:
            cmap_fit = get_discrete_cmap("fitness_map", n=24)
        cmap_fit, _fit_norm, boundaries = build_fitness_quantile_colormap(
            fitness_concat, n_bins=fitness_bins, cmap_name_or_obj=cmap_fit
        )
        colorscale_fit = mpl_cmap_to_plotly_scale(cmap_fit)
        bin_centers = [0.5 * (boundaries[i] + boundaries[i + 1]) for i in range(len(boundaries) - 1)]
        bin_idx = np.digitize(fitness_concat, boundaries[1:-1], right=False)
        bin_idx = np.clip(bin_idx, 0, fitness_bins - 1)
        tick_vals = list(range(fitness_bins))
        tick_text = [f"{v:.2f}" for v in bin_centers]

        fig.add_trace(
            go.Scattergl(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode="markers",
                marker=dict(
                    size=7,
                    color=bin_idx,
                    colorscale=colorscale_fit,
                    colorbar=dict(
                        title="Fitness",
                        tickvals=tick_vals,
                        ticktext=tick_text,
                        x=1.03,
                    ),
                    cmin=0,
                    cmax=fitness_bins - 1,
                    opacity=0.8,
                ),
                hovertemplate="Fitness bin %{marker.color}<br>Gen %{customdata[0]}<extra></extra>"
                if customdata is not None
                else "Fitness bin %{marker.color}<extra></extra>",
                customdata=customdata,
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title="UMAP-1")
    fig.update_yaxes(title="UMAP-2")
    fig.update_layout(
        height=520,
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig
