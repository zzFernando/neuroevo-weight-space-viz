"""Trajectory bundling with smoother curves, adaptive transparency, clustering, and best-trajectory highlight."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from utils import compute_aligned_umap_embedding
from .common import build_fitness_quantile_colormap, mpl_cmap_to_plotly_scale


def _moving_average(traj: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1:
        return traj
    padded = np.pad(traj, ((window // 2, window // 2), (0, 0)), mode="edge")
    cumsum = np.cumsum(padded, axis=0)
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    return smoothed


def _catmull_rom_spline(points: np.ndarray, n_points: int) -> np.ndarray:
    if len(points) < 4:
        return points
    t = np.linspace(0, len(points) - 1, n_points)
    t0 = np.floor(t).astype(int)
    t1 = np.clip(t0 + 1, 0, len(points) - 1)
    t_ = t - t0

    p0 = points[np.clip(t0 - 1, 0, len(points) - 1)]
    p1 = points[t0]
    p2 = points[t1]
    p3 = points[np.clip(t1 + 1, 0, len(points) - 1)]

    a = 2 * p1
    b = p2 - p0
    c = 2 * p0 - 5 * p1 + 4 * p2 - p3
    d = -p0 + 3 * p1 - 3 * p2 + p3

    t2 = t_ ** 2
    t3 = t_ ** 3
    return 0.5 * (a + b * t_[:, None] + c * t2[:, None] + d * t3[:, None])


def _density_alpha(points: np.ndarray, base_alpha: float = 0.4, k: int = 15) -> np.ndarray:
    if len(points) == 0:
        return np.array([])
    nbrs = NearestNeighbors(n_neighbors=min(k, len(points))).fit(points)
    dist, _ = nbrs.kneighbors(points)
    mean_dist = dist[:, 1:].mean(axis=1)
    norm = (mean_dist - mean_dist.min()) / (np.ptp(mean_dist) + 1e-8)
    return base_alpha * (0.2 + 0.8 * norm)


@dataclass
class TrajectoryBundler:
    beta: float = 0.25
    iterations: int = 25
    neighbor_radius: float | None = None
    resample_points: int = 30
    temporal_smooth: int = 3
    curve_type: str = "catmull-rom"  # catmull-rom | polyline

    def resample(self, traj: np.ndarray) -> np.ndarray:
        t_old = np.linspace(0.0, 1.0, len(traj))
        t_new = np.linspace(0.0, 1.0, self.resample_points)
        x = np.interp(t_new, t_old, traj[:, 0])
        y = np.interp(t_new, t_old, traj[:, 1])
        return np.stack([x, y], axis=1)

    def bundle(self, control_points: np.ndarray) -> np.ndarray:
        bundled = control_points.copy()
        n_traj, n_ctrl, _ = bundled.shape

        for _ in range(self.iterations):
            for idx_ctrl in range(1, n_ctrl - 1):
                pts = bundled[:, idx_ctrl, :]
                updates = np.zeros_like(pts)
                for i, p in enumerate(pts):
                    diffs = pts - p
                    dist = np.linalg.norm(diffs, axis=1)
                    mask = (dist > 0) & (dist < (self.neighbor_radius or np.inf))
                    if np.any(mask):
                        updates[i] = self.beta * diffs[mask].mean(axis=0)
                bundled[:, idx_ctrl, :] = pts + updates
        return bundled

    def smooth_curve(self, traj: np.ndarray) -> np.ndarray:
        if self.curve_type == "catmull-rom":
            return _catmull_rom_spline(traj, len(traj) * 2)
        return traj


def _prepare_bundled_data(
    weights_by_gen: Sequence[np.ndarray],
    fitness_by_gen: Sequence[np.ndarray],
    lambda_align: float,
    beta: float,
    iterations: int,
    resample_points: int,
    neighbor_radius: float | None,
    random_state: int,
    temporal_smooth: int,
    curve_type: str,
    n_clusters: int,
    max_trajectories: int | None,
    show_representatives_only: bool = False,
    show_top_k: bool = False,
    top_k: int | None = None,
):
    _, _, per_gen_embeddings = compute_aligned_umap_embedding(
        weights_by_gen, lambda_align=lambda_align, random_state=random_state
    )

    n_steps = len(per_gen_embeddings)
    if n_steps < 2:
        raise ValueError("At least two generations are required for trajectory bundling.")

    n_traj = min(len(gen) for gen in per_gen_embeddings)
    if max_trajectories:
        n_traj = min(n_traj, max_trajectories)
    if n_traj == 0:
        raise ValueError("No individuals available to build trajectories.")

    # Seleciona pela fitness final (última geração)
    fitness_last = fitness_by_gen[-1]
    order = np.argsort(fitness_last)[::-1]
    keep_idx = order[:n_traj]

    trajectories = np.stack([[per_gen_embeddings[g][i] for g in range(n_steps)] for i in keep_idx])

    bundler = TrajectoryBundler(
        beta=beta,
        iterations=iterations,
        neighbor_radius=neighbor_radius,
        resample_points=resample_points,
        temporal_smooth=temporal_smooth,
        curve_type=curve_type,
    )

    control_points = np.stack([bundler.resample(traj) for traj in trajectories])
    if bundler.temporal_smooth > 1:
        control_points = np.stack([_moving_average(traj, bundler.temporal_smooth) for traj in control_points])

    if bundler.neighbor_radius is None:
        all_pts = control_points.reshape(-1, 2)
        diag = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))
        bundler.neighbor_radius = 0.08 * diag

    bundled = bundler.bundle(control_points)
    bundled_smooth = np.stack([bundler.smooth_curve(traj) for traj in bundled])

    cluster_labels = None
    if n_clusters and n_clusters > 1 and n_traj >= n_clusters:
        features = bundled_smooth.reshape(n_traj, -1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(features)

    fitness_matrix = np.vstack([f[keep_idx] for f in fitness_by_gen])
    traj_fitness_final = fitness_matrix[-1]  # fitness final por trajetória selecionada

    # Representatives: pick one trajectory per cluster (highest fitness in cluster)
    if show_representatives_only and cluster_labels is not None:
        reps = []
        rep_labels = []
        rep_fitness = []
        for cl in np.unique(cluster_labels):
            idx_in_cl = np.where(cluster_labels == cl)[0]
            if len(idx_in_cl):
                best_local = idx_in_cl[np.argmax(traj_fitness_final[idx_in_cl])]
                reps.append(bundled_smooth[best_local])
                rep_labels.append(cluster_labels[best_local])
                rep_fitness.append(traj_fitness_final[best_local])
        bundled_smooth = np.stack(reps) if reps else bundled_smooth
        cluster_labels = np.array(rep_labels) if reps else cluster_labels
        traj_fitness = np.array(rep_fitness) if reps else traj_fitness

    # Optionally keep only top-k trajectories by fitness
    if show_top_k and top_k:
        top_k = min(top_k, len(traj_fitness))
        order_top = np.argsort(traj_fitness)[::-1][:top_k]
        bundled_smooth = bundled_smooth[order_top]
        traj_fitness_final = traj_fitness_final[order_top]
        if cluster_labels is not None:
            cluster_labels = cluster_labels[order_top]

    return bundled_smooth, traj_fitness_final, cluster_labels


def plot(
    weights_by_gen: Sequence[np.ndarray],
    fitness_by_gen: Sequence[np.ndarray],
    lambda_align: float = 0.3,
    beta: float = 0.25,
    iterations: int = 25,
    resample_points: int = 30,
    neighbor_radius: float | None = None,
    random_state: int = 42,
    highlight_best: bool = True,
    cmap=plt.cm.plasma,
    line_alpha: float = 0.4,
    line_width: float = 1.3,
    n_clusters: int = 3,
    curve_type: str = "catmull-rom",
    temporal_smooth: int = 3,
    norm_mode: str = "linear",
    max_trajectories: int | None = None,
    gamma: float = 0.3,
    fitness_bins: int = 9,
    show_representatives_only: bool = False,
    show_top_k: bool = False,
    top_k: int | None = None,
    color_clip_quantiles: tuple[float, float] = (0.05, 0.95),
    colorbar_label: str = "Final fitness (higher is better)",
    show_best: bool = True,
    best_label: str = "Best trajectory (highest final fitness)",
    color_lines_by_fitness: bool = False,
):
    """
    Plot bundled trajectories using aligned UMAP coordinates across generations.
    Includes clustering, adaptive transparency, and smoothing.
    """
    show_best = show_best and highlight_best

    bundled_smooth, traj_fitness, cluster_labels = _prepare_bundled_data(
        weights_by_gen,
        fitness_by_gen,
        lambda_align,
        beta,
        iterations,
        resample_points,
        neighbor_radius,
        random_state,
        temporal_smooth,
        curve_type,
        n_clusters,
        max_trajectories,
        show_representatives_only,
        show_top_k,
        top_k,
    )
    q_lo, q_hi = color_clip_quantiles
    lo, hi = np.quantile(traj_fitness, [q_lo, q_hi]) if len(traj_fitness) else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1e-6
    norm = mcolors.Normalize(vmin=lo, vmax=hi, clip=True)

    alphas = _density_alpha(bundled_smooth.reshape(-1, 2), base_alpha=line_alpha)
    alpha_min, alpha_max = alphas.min(), alphas.max()

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    cmap_clusters = plt.cm.tab10

    for i, traj in enumerate(bundled_smooth):
        traj_alpha = np.linspace(alpha_max, alpha_min, len(traj))
        if color_lines_by_fitness:
            color_val = cmap(norm(traj_fitness[i])) if cluster_labels is None else cmap_clusters(cluster_labels[i] % cmap_clusters.N)
            lw = line_width
            alpha_line = traj_alpha.mean()
        else:
            color_val = "#B0B0B0"
            lw = max(line_width * 0.8, 0.6)
            alpha_line = 0.2
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=color_val,
            alpha=alpha_line,
            linewidth=lw,
        )

    if show_best and len(traj_fitness):
        best_idx = int(np.argmax(traj_fitness))
        ax.plot(
            bundled_smooth[best_idx, :, 0],
            bundled_smooth[best_idx, :, 1],
            color="black",
            linewidth=line_width + 2.0,
            alpha=1.0,
            label=best_label,
        )
        ax.scatter(
            bundled_smooth[best_idx, 0, 0],
            bundled_smooth[best_idx, 0, 1],
            color="black",
            edgecolors="white",
            s=60,
            zorder=5,
            label="_start",
        )
        ax.scatter(
            bundled_smooth[best_idx, -1, 0],
            bundled_smooth[best_idx, -1, 1],
            color="white",
            edgecolors="black",
            s=70,
            marker="^",
            zorder=6,
            label="_end",
        )
        ax.legend(loc="upper right", frameon=False, fontsize=9)

    # Colorbar from endpoints (fitness final)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.18, fraction=0.08)
    cbar.set_label(colorbar_label)
    cbar.ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    # Endpoints colored by final fitness
    if len(bundled_smooth):
        endpoints = bundled_smooth[:, -1, :]
        sc = ax.scatter(
            endpoints[:, 0],
            endpoints[:, 1],
            c=traj_fitness,
            cmap=cmap,
            norm=norm,
            s=16,
            alpha=0.9,
            edgecolors="none",
            zorder=4,
        )

    ax.set_title("Trajectory Bundling", fontsize=13)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.15, linestyle="--", linewidth=0.4)
    for spine in ax.spines.values():
        spine.set_alpha(0.2)
    return fig


def plot_interactive(
    weights_by_gen: Sequence[np.ndarray],
    fitness_by_gen: Sequence[np.ndarray],
    lambda_align: float = 0.3,
    beta: float = 0.25,
    iterations: int = 25,
    resample_points: int = 30,
    neighbor_radius: float | None = None,
    random_state: int = 42,
    highlight_best: bool = True,
    cmap=plt.cm.plasma,
    line_alpha: float = 0.55,
    line_width: float = 2.0,
    n_clusters: int = 3,
    curve_type: str = "catmull-rom",
    temporal_smooth: int = 3,
    norm_mode: str = "linear",
    max_trajectories: int | None = None,
    gamma: float = 0.3,
    fitness_bins: int = 9,
    show_representatives_only: bool = False,
    show_top_k: bool = False,
    top_k: int | None = None,
    color_clip_quantiles: tuple[float, float] = (0.05, 0.95),
    colorbar_label: str = "Final fitness (higher is better)",
    show_best: bool = True,
    best_label: str = "Best trajectory (highest final fitness)",
    color_lines_by_fitness: bool = False,
):
    """
    Interactive Plotly variant of the trajectory bundling visualization.
    """
    _ = (norm_mode, gamma)  # kept for API compatibility with the static version
    show_best = show_best and highlight_best

    bundled_smooth, traj_fitness, cluster_labels = _prepare_bundled_data(
        weights_by_gen,
        fitness_by_gen,
        lambda_align,
        beta,
        iterations,
        resample_points,
        neighbor_radius,
        random_state,
        temporal_smooth,
        curve_type,
        n_clusters,
        max_trajectories,
        show_representatives_only,
        show_top_k,
        top_k,
    )

    q_lo, q_hi = color_clip_quantiles
    lo, hi = np.quantile(traj_fitness, [q_lo, q_hi]) if len(traj_fitness) else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1e-6
    colorscale_fit = mpl_cmap_to_plotly_scale(cmap)
    cmap_clusters = plt.cm.tab10

    fig = go.Figure()

    # Draw trajectories in gray (unless coloring is requested)
    for i, traj in enumerate(bundled_smooth):
        customdata = np.column_stack(
            [np.full(len(traj), i, dtype=int), np.full(len(traj), traj_fitness[i], dtype=float)]
        )
        line_color = (
            mcolors.to_hex(cmap((traj_fitness[i] - lo) / (hi - lo)))
            if color_lines_by_fitness and cluster_labels is None
            else mcolors.to_hex(cmap_clusters(cluster_labels[i] % cmap_clusters.N))
            if color_lines_by_fitness and cluster_labels is not None
            else "#B0B0B0"
        )
        fig.add_trace(
            go.Scatter(
                x=traj[:, 0],
                y=traj[:, 1],
                mode="lines",
                line=dict(color=line_color, width=line_width if color_lines_by_fitness else max(line_width * 0.8, 0.6)),
                opacity=line_alpha if color_lines_by_fitness else 0.2,
                hovertemplate="Trajectory %{customdata[0]}<br>Final fitness %{customdata[1]:.3f}"
                "<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>",
                customdata=customdata,
                showlegend=False,
            )
        )

    # Color endpoints by final fitness
    if len(bundled_smooth):
        endpoints = bundled_smooth[:, -1, :]
        fig.add_trace(
            go.Scatter(
                x=endpoints[:, 0],
                y=endpoints[:, 1],
                mode="markers",
                marker=dict(
                    size=16,
                    color=traj_fitness,
                    colorscale=colorscale_fit,
                    cmin=lo,
                    cmax=hi,
                    colorbar=dict(title=colorbar_label),
                    opacity=0.9,
                    line=dict(width=0),
                ),
                hovertemplate="Final fitness %{marker.color:.3f}<extra></extra>",
                showlegend=False,
            )
        )

    if show_best and len(traj_fitness):
        best_idx = int(np.argmax(traj_fitness))
        fig.add_trace(
            go.Scatter(
                x=bundled_smooth[best_idx, :, 0],
                y=bundled_smooth[best_idx, :, 1],
                mode="lines",
                line=dict(color="black", width=line_width + 1.8),
                opacity=1.0,
                name=best_label,
                hovertemplate=f"{best_label}<br>Fitness {traj_fitness[best_idx]:.3f}<extra></extra>",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[bundled_smooth[best_idx, -1, 0]],
                y=[bundled_smooth[best_idx, -1, 1]],
                mode="markers",
                marker=dict(
                    size=18,
                    color=traj_fitness[best_idx],
                    colorscale=colorscale_fit,
                    cmin=lo,
                    cmax=hi,
                    line=dict(color="white", width=1.2),
                ),
                hovertemplate=f"{best_label}<br>Final fitness {traj_fitness[best_idx]:.3f}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_xaxes(title="UMAP-1")
    fig.update_yaxes(title="UMAP-2")
    fig.update_layout(
        title="Trajectory Bundling (interactive)",
        height=560,
        margin=dict(l=40, r=20, t=60, b=40),
        showlegend=show_best,
    )
    return fig
