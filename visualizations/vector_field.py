"""
Vector Field visualization following Cantareira et al. (2020).
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from utils import compute_aligned_umap_embedding


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


def plot(
    weights_by_gen: Sequence[np.ndarray],
    fitness_by_gen: Sequence[np.ndarray],
    lambda_align: float = 0.3,
    grid_res: int = 20,
    random_state: int = 42,
    show_points: bool = True,
):
    """
    Plot streamlines of the average displacement field between consecutive
    generations in the aligned UMAP space.
    """
    embedding, gen_labels, per_gen_embeddings = compute_aligned_umap_embedding(
        weights_by_gen, lambda_align=lambda_align, random_state=random_state
    )

    Xc, Yc, U, V, speed = _compute_velocity_grid(per_gen_embeddings, grid_res=grid_res)

    fig, ax = plt.subplots(figsize=(8, 6))

    if show_points:
        gens = np.unique(gen_labels)
        cmap_gens = plt.cm.get_cmap("tab20", max(len(gens), 1))
        for idx, g in enumerate(gens):
            pts = embedding[gen_labels == g]
            ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.35, color=cmap_gens(idx), label=f"G{g}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    strm = ax.streamplot(
        Xc,
        Yc,
        U,
        V,
        density=2.0,
        color=speed,
        cmap="magma",
        linewidth=1.2,
    )
    cbar = fig.colorbar(strm.lines, ax=ax)
    cbar.set_label("Velocity magnitude")

    ax.set_title("Vector Field of Representation Flow (Cantareira 2020)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    return fig
