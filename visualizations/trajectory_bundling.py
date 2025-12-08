"""
Trajectory bundling inspired by Cantareira et al. (2020), with smoother curves,
adaptive transparency, clustering, and highlight of the best trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from utils import compute_aligned_umap_embedding
from .common import AdaptiveColorNorm, build_fitness_quantile_colormap


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
):
    """
    Plot bundled trajectories using aligned UMAP coordinates across generations.
    Includes clustering, adaptive transparency, and smoothing.
    """
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

    # select top trajectories by mean fitness if limiting
    fitness_matrix_full = np.vstack([f[: min(len(f), n_traj * 2)] for f in fitness_by_gen])
    mean_fit_full = fitness_matrix_full.mean(axis=0)
    order = np.argsort(mean_fit_full)[::-1]
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

    # Clustering trajectories for color grouping
    cluster_labels = None
    if n_clusters and n_clusters > 1 and n_traj >= n_clusters:
        features = bundled_smooth.reshape(n_traj, -1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(features)

    fitness_matrix = np.vstack([f[keep_idx] for f in fitness_by_gen])
    traj_fitness = fitness_matrix.mean(axis=0)
    cmap, fit_norm, boundaries = build_fitness_quantile_colormap(
        traj_fitness, n_bins=fitness_bins, cmap_name=cmap.name if hasattr(cmap, "name") else "plasma"
    )

    alphas = _density_alpha(bundled_smooth.reshape(-1, 2), base_alpha=line_alpha)
    alpha_min, alpha_max = alphas.min(), alphas.max()

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap_clusters = plt.cm.tab10

    for i, traj in enumerate(bundled_smooth):
        traj_alpha = np.linspace(alpha_max, alpha_min, len(traj))
        color_val = cmap(fit_norm(traj_fitness[i])) if cluster_labels is None else cmap_clusters(cluster_labels[i] % cmap_clusters.N)
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=color_val,
            alpha=traj_alpha.mean(),
            linewidth=line_width,
        )

    if highlight_best and len(traj_fitness):
        best_idx = int(np.argmax(traj_fitness))
        ax.plot(
            bundled_smooth[best_idx, :, 0],
            bundled_smooth[best_idx, :, 1],
            color="white",
            linewidth=2.6,
            alpha=0.95,
            label="Best trajectory",
        )
        ax.legend(loc="upper right", frameon=False, fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=fit_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.12, fraction=0.05)
    cbar.set_label("Mean fitness along trajectory")

    ax.set_title("Trajectory Bundling (Cantareira 2020)", fontsize=13)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    return fig
