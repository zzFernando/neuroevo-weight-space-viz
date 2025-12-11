"""Vector Field visualization."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import compute_aligned_umap_embedding
from .common import build_fitness_quantile_colormap, mpl_cmap_to_plotly_scale, scatter_2d


def _compute_velocity_grid(per_gen_embeddings: Sequence[np.ndarray], grid_res: int):
    if len(per_gen_embeddings) < 2:
        raise ValueError("At least two generations are required to compute a vector field.")

    # Bounding box across all generations
    stacked = np.vstack(per_gen_embeddings)
    x_min, x_max = stacked[:, 0].min(), stacked[:, 0].max()
    y_min, y_max = stacked[:, 1].min(), stacked[:, 1].max()

    edges_x = np.linspace(x_min, x_max, grid_res + 1)
    edges_y = np.linspace(y_min, y_max, grid_res + 1)
    centers_x = (edges_x[:-1] + edges_x[1:]) / 2
    centers_y = (edges_y[:-1] + edges_y[1:]) / 2

    U = np.zeros((grid_res, grid_res))
    V = np.zeros((grid_res, grid_res))
    count = np.zeros((grid_res, grid_res), dtype=int)

    n_traj = min(len(gen) for gen in per_gen_embeddings)

    for g in range(len(per_gen_embeddings) - 1):
        P = per_gen_embeddings[g][:n_traj]
        Q = per_gen_embeddings[g + 1][:n_traj]
        velocities = Q - P

        for p, vel in zip(P, velocities):
            x, y = p
            ix = np.searchsorted(edges_x, x) - 1
            iy = np.searchsorted(edges_y, y) - 1
            if 0 <= ix < grid_res and 0 <= iy < grid_res:
                U[iy, ix] += vel[0]
                V[iy, ix] += vel[1]
                count[iy, ix] += 1

    mask = count == 0
    U_mean = np.zeros_like(U)
    V_mean = np.zeros_like(V)
    U_mean[~mask] = U[~mask] / count[~mask]
    V_mean[~mask] = V[~mask] / count[~mask]

    speed = np.sqrt(U_mean ** 2 + V_mean ** 2)

    Xc, Yc = np.meshgrid(centers_x, centers_y)
    return Xc, Yc, np.ma.array(U_mean, mask=mask), np.ma.array(V_mean, mask=mask), speed


def _gaussian_kernel(sigma: float) -> np.ndarray:
    half_size = max(1, int(3 * sigma))
    x = np.arange(-half_size, half_size + 1)
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return np.outer(g, g)


def _gaussian_smooth(mat: np.ma.MaskedArray, sigma: float) -> np.ma.MaskedArray:
    if sigma <= 0:
        return mat
    kernel = _gaussian_kernel(sigma)
    pad = kernel.shape[0] // 2
    data = np.ma.filled(mat, 0.0)
    padded = np.pad(data, pad, mode="edge")
    out = np.zeros_like(data)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            window = padded[i : i + kernel.shape[0], j : j + kernel.shape[1]]
            out[i, j] = np.sum(window * kernel)
    return np.ma.array(out, mask=mat.mask)


def _draw_vectors(ax, X, Y, U, V, speed, mode: str, cmap):
    if mode == "quiver":
        return ax.quiver(X, Y, U, V, speed, cmap=cmap, scale=40, width=0.004)
    return ax.streamplot(
        X,
        Y,
        U,
        V,
        density=1.2,
        color=speed,
        cmap=cmap,
        linewidth=1.3,
    )


def _prepare_field_data(
    weights_by_gen: Sequence[np.ndarray],
    fitness_by_gen: Sequence[np.ndarray],
    lambda_align: float,
    grid_res: int,
    random_state: int,
    normalize_vectors: bool,
    smoothing_sigma: float,
    subsample: int,
    quantize_bins: int | None,
):
    embedding, gen_labels, per_gen_embeddings = compute_aligned_umap_embedding(
        weights_by_gen, lambda_align=lambda_align, random_state=random_state
    )
    Xc, Yc, U, V, speed = _compute_velocity_grid(per_gen_embeddings, grid_res=grid_res)

    if normalize_vectors:
        mag = np.sqrt(U ** 2 + V ** 2)
        U = np.ma.array(np.where(mag > 0, U / mag, 0), mask=U.mask)
        V = np.ma.array(np.where(mag > 0, V / mag, 0), mask=V.mask)
    if smoothing_sigma > 0:
        U = _gaussian_smooth(U, sigma=smoothing_sigma)
        V = _gaussian_smooth(V, sigma=smoothing_sigma)
        speed = np.sqrt(U ** 2 + V ** 2)

    if subsample > 1:
        U = U[::subsample, ::subsample]
        V = V[::subsample, ::subsample]
        Xc = Xc[::subsample, ::subsample]
        Yc = Yc[::subsample, ::subsample]
        speed = speed[::subsample, ::subsample]

    if quantize_bins and quantize_bins > 1:
        quantized = np.linspace(speed.min(), speed.max(), quantize_bins)
        speed = np.digitize(speed, quantized)

    fitness_concat = np.concatenate(fitness_by_gen) if fitness_by_gen else np.array([])
    return embedding, gen_labels, fitness_concat, Xc, Yc, U, V, speed


def plot(
    weights_by_gen: Sequence[np.ndarray],
    fitness_by_gen: Sequence[np.ndarray],
    lambda_align: float = 0.3,
    grid_res: int = 20,
    random_state: int = 42,
    show_points: bool = True,
    normalize_vectors: bool = False,
    smoothing_sigma: float = 1.0,
    subsample: int = 1,
    vector_mode: str = "stream",  # stream or quiver
    quantize_bins: int | None = None,
    cmap_gen=plt.cm.plasma,
    cmap_fit=plt.cm.cividis,
    cmap_speed=plt.cm.magma,
    norm_mode: str = "power",
    gamma: float = 0.3,
    fitness_bins: int = 9,
):
    """
    Plot streamlines of the average displacement field between consecutive
    generations in the aligned UMAP space. Shows two panels:
    - left: points colored by generation (continuous gradient)
    - right: points colored by fitness
    """
    (
        embedding,
        gen_labels,
        fitness_concat,
        Xc,
        Yc,
        U,
        V,
        speed,
    ) = _prepare_field_data(
        weights_by_gen,
        fitness_by_gen,
        lambda_align,
        grid_res,
        random_state,
        normalize_vectors,
        smoothing_sigma,
        subsample,
        quantize_bins,
    )
    cmap_fit, fit_norm, boundaries = build_fitness_quantile_colormap(
        fitness_concat, n_bins=fitness_bins, cmap_name=cmap_fit.name if hasattr(cmap_fit, "name") else "plasma"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    # Left: colored by generation
    ax = axes[0]
    if show_points:
        norm_gen = plt.Normalize(vmin=gen_labels.min(), vmax=gen_labels.max() if len(gen_labels) else 1)
        sc_gen = scatter_2d(ax, embedding, gen_labels, cmap_gen, norm_gen, size=10, base_alpha=0.4)
        cbar_gen = fig.colorbar(sc_gen, ax=ax, orientation="horizontal", pad=0.12, fraction=0.05)
        cbar_gen.set_label("Generation index")

    strm_left = _draw_vectors(ax, Xc, Yc, U, V, speed, mode=vector_mode, cmap=cmap_speed)
    ax.set_title("Vector Field — Generation", fontsize=13)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)

    # Right: colored by fitness
    ax = axes[1]
    if show_points:
        sc = scatter_2d(ax, embedding, fitness_concat, cmap_fit, fit_norm, size=10, base_alpha=0.5)
        cbar_fit = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.12, fraction=0.05)
        cbar_fit.set_label("Fitness (quantile bins)")
        tick_positions = np.linspace(0, len(boundaries) - 2, 5, dtype=int)
        tick_values = [0.5 * (boundaries[i] + boundaries[i + 1]) for i in tick_positions]
        cbar_fit.set_ticks(tick_values)
        cbar_fit.ax.set_xticklabels([f"{v:.2f}" for v in tick_values])

    strm_right = _draw_vectors(ax, Xc, Yc, U, V, speed, mode=vector_mode, cmap=cmap_speed)
    ax.set_title("Vector Field — Fitness", fontsize=13)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)

    speed_artist = strm_right if vector_mode == "quiver" else strm_right.lines
    cbar_speed = fig.colorbar(speed_artist, ax=axes, fraction=0.035, pad=0.02)
    cbar_speed.set_label("Velocity magnitude")

    fig.tight_layout()
    return fig


def plot_interactive(
    weights_by_gen: Sequence[np.ndarray],
    fitness_by_gen: Sequence[np.ndarray],
    lambda_align: float = 0.3,
    grid_res: int = 20,
    random_state: int = 42,
    show_points: bool = True,
    normalize_vectors: bool = False,
    smoothing_sigma: float = 1.0,
    subsample: int = 1,
    vector_mode: str = "stream",  # kept for API parity, arrows are quiver-like here
    quantize_bins: int | None = None,
    cmap_gen=plt.cm.plasma,
    cmap_fit=plt.cm.cividis,
    cmap_speed=plt.cm.magma,
    norm_mode: str = "power",
    gamma: float = 0.3,
    fitness_bins: int = 9,
):
    """
    Interactive Plotly variant of the vector field visualization.
    """
    _ = (vector_mode, norm_mode, gamma)  # kept for signature compatibility; arrows are rendered as quivers
    (
        embedding,
        gen_labels,
        fitness_concat,
        Xc,
        Yc,
        U,
        V,
        speed,
    ) = _prepare_field_data(
        weights_by_gen,
        fitness_by_gen,
        lambda_align,
        grid_res,
        random_state,
        normalize_vectors,
        smoothing_sigma,
        subsample,
        quantize_bins,
    )

    colorscale_gen = mpl_cmap_to_plotly_scale(cmap_gen)
    cmap_fit, _fit_norm, boundaries = build_fitness_quantile_colormap(
        fitness_concat, n_bins=fitness_bins, cmap_name=cmap_fit.name if hasattr(cmap_fit, "name") else "plasma"
    )
    colorscale_fit = mpl_cmap_to_plotly_scale(cmap_fit)
    colorscale_speed = mpl_cmap_to_plotly_scale(cmap_speed)

    speed_img = np.ma.filled(speed, np.nan)
    speed_min = float(np.nanmin(speed_img)) if np.any(~np.isnan(speed_img)) else 0.0
    speed_max = float(np.nanmax(speed_img)) if np.any(~np.isnan(speed_img)) else 1.0

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Vector Field — Generation", "Vector Field — Fitness"],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )

    def add_vector_panel(col: int, show_colorbar: bool):
        fig.add_trace(
            go.Heatmap(
                x=Xc[0],
                y=Yc[:, 0],
                z=speed_img,
                colorscale=colorscale_speed,
                showscale=show_colorbar,
                colorbar=dict(title="Velocity magnitude") if show_colorbar else None,
                opacity=0.55,
                zmin=speed_min,
                zmax=speed_max,
                hovertemplate="Speed %{z:.4f}<extra></extra>",
            ),
            row=1,
            col=col,
        )

        valid = ~U.mask if isinstance(U, np.ma.MaskedArray) else np.ones_like(U, dtype=bool)
        X_flat = Xc[valid]
        Y_flat = Yc[valid]
        U_flat = U[valid]
        V_flat = V[valid]

        if X_flat.size:
            quiv = ff.create_quiver(
                X_flat,
                Y_flat,
                U_flat,
                V_flat,
                scale=0.2,
                arrow_scale=0.25,
                line_color="black",
                name="flow",
            )
            for trace in quiv.data:
                trace.showlegend = False
                fig.add_trace(trace, row=1, col=col)

    add_vector_panel(col=1, show_colorbar=False)
    add_vector_panel(col=2, show_colorbar=True)

    if len(embedding) and show_points:
        customdata = (
            np.stack([gen_labels, fitness_concat], axis=1) if len(fitness_concat) == len(gen_labels) else None
        )
        gen_min, gen_max = gen_labels.min(), gen_labels.max()
        tick_positions = np.linspace(0, len(boundaries) - 2, 5, dtype=int)
        tick_values = [0.5 * (boundaries[i] + boundaries[i + 1]) for i in tick_positions]

        fig.add_trace(
            go.Scattergl(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode="markers",
                marker=dict(
                    size=6,
                    color=gen_labels,
                    colorscale=colorscale_gen,
                    colorbar=dict(title="Generation index"),
                    cmin=gen_min,
                    cmax=gen_max,
                    opacity=0.7,
                ),
                hovertemplate="Gen %{customdata[0]}<br>Fitness %{customdata[1]:.3f}<extra></extra>"
                if customdata is not None
                else "Gen %{marker.color}<extra></extra>",
                customdata=customdata,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scattergl(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode="markers",
                marker=dict(
                    size=6,
                    color=fitness_concat,
                    colorscale=colorscale_fit,
                    colorbar=dict(
                        title="Fitness (quantile bins)",
                        tickvals=tick_values,
                        ticktext=[f"{v:.2f}" for v in tick_values],
                    ),
                    cmin=boundaries[0],
                    cmax=boundaries[-1],
                    opacity=0.7,
                ),
                hovertemplate="Fitness %{marker.color:.3f}<br>Gen %{customdata[0]}<extra></extra>"
                if customdata is not None
                else "Fitness %{marker.color:.3f}<extra></extra>",
                customdata=customdata,
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title="UMAP-1")
    fig.update_yaxes(title="UMAP-2")
    fig.update_layout(height=540, showlegend=False, margin=dict(l=40, r=20, t=60, b=40))
    return fig
