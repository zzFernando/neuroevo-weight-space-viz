"""
Trajectory bundling inspired by Cantareira et al. (2020) to reduce visual clutter.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from utils import compute_aligned_umap_embedding


def _resample(traj: np.ndarray, num_points: int) -> np.ndarray:
    t_old = np.linspace(0.0, 1.0, len(traj))
    t_new = np.linspace(0.0, 1.0, num_points)
    x = np.interp(t_new, t_old, traj[:, 0])
    y = np.interp(t_new, t_old, traj[:, 1])
    return np.stack([x, y], axis=1)


def _bundle(control_points: np.ndarray, beta: float, iterations: int, radius: float) -> np.ndarray:
    bundled = control_points.copy()
    n_traj, n_ctrl, _ = bundled.shape

    for _ in range(iterations):
        for idx_ctrl in range(1, n_ctrl - 1):  # keep endpoints fixed
            pts = bundled[:, idx_ctrl, :]
            updates = np.zeros_like(pts)
            for i, p in enumerate(pts):
                diffs = pts - p
                dist = np.linalg.norm(diffs, axis=1)
                mask = (dist > 0) & (dist < radius)
                if np.any(mask):
                    updates[i] = beta * diffs[mask].mean(axis=0)
            bundled[:, idx_ctrl, :] = pts + updates
    return bundled


def plot(
    weights_by_gen: Sequence[np.ndarray],
    fitness_by_gen: Sequence[np.ndarray],
    lambda_align: float = 0.3,
    beta: float = 0.35,
    iterations: int = 15,
    resample_points: int = 20,
    neighbor_radius: float | None = None,
    random_state: int = 42,
):
    """
    Plot bundled trajectories using aligned UMAP coordinates across generations.
    """
    _, _, per_gen_embeddings = compute_aligned_umap_embedding(
        weights_by_gen, lambda_align=lambda_align, random_state=random_state
    )

    n_steps = len(per_gen_embeddings)
    if n_steps < 2:
        raise ValueError("At least two generations are required for trajectory bundling.")

    n_traj = min(len(gen) for gen in per_gen_embeddings)
    if n_traj == 0:
        raise ValueError("No individuals available to build trajectories.")

    trajectories = np.stack(
        [[per_gen_embeddings[g][i] for g in range(n_steps)] for i in range(n_traj)]
    )

    control_points = np.stack([_resample(traj, resample_points) for traj in trajectories])

    if neighbor_radius is None:
        all_pts = control_points.reshape(-1, 2)
        diag = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))
        neighbor_radius = 0.1 * diag

    bundled = _bundle(control_points, beta=beta, iterations=iterations, radius=neighbor_radius)

    fitness_matrix = np.vstack([f[:n_traj] for f in fitness_by_gen])
    traj_fitness = fitness_matrix.mean(axis=0)
    norm = plt.Normalize(vmin=traj_fitness.min(), vmax=traj_fitness.max())
    cmap = plt.cm.plasma

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, traj in enumerate(bundled):
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=cmap(norm(traj_fitness[i])),
            alpha=0.4,
            linewidth=2.0,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Mean fitness along trajectory")

    ax.set_title("Trajectory Bundling (Cantareira 2020)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    return fig
