"""Visualization utilities tailored to neuroevolution weight dynamics."""
from __future__ import annotations

from typing import Sequence
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _as_matrix(history: Sequence[np.ndarray]) -> np.ndarray:
    if history is None or len(history) == 0:
        raise ValueError("history is empty.")
    return np.vstack(history)


def plot_weight_heatmap(best_history: Sequence[np.ndarray],
                        save_path: str,
                        max_weights: int = 256) -> None:
    """2-D heatmap with generations on X and variable index on Y."""
    matrix = _as_matrix(best_history)
    if matrix.shape[1] > max_weights:
        variances = matrix.var(axis=0)
        idx = np.argsort(variances)[-max_weights:]
        idx.sort()
        matrix = matrix[:, idx]

    data = matrix.T  # rows: variables, cols: generations
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, aspect="auto", cmap="coolwarm", interpolation="nearest",
                   origin="lower", extent=[0, data.shape[1] - 1, 0, data.shape[0] - 1])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Variable index")
    ax.set_title("Heatmap of best-individual weights over generations")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Weight value")
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_best_mds(best_history: Sequence[np.ndarray],
                  save_path: str,
                  random_state: int = 0) -> None:
    """Map weight vectors into 2-D via MDS to visualize search trajectory."""
    matrix = _as_matrix(best_history)
    if matrix.shape[0] < 2:
        raise ValueError("Need at least two generations for MDS plot.")

    embedding = MDS(n_components=2, random_state=random_state, dissimilarity="euclidean")
    coords = embedding.fit_transform(matrix)
    generations = np.arange(matrix.shape[0])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(coords[:, 0], coords[:, 1], color="#bbbbbb", linewidth=1.2, alpha=0.6)
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=generations,
        cmap="viridis",
        s=60,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.scatter(coords[0, 0], coords[0, 1], marker="^", s=140,
               color="#1b9e77", label="Start")
    ax.scatter(coords[-1, 0], coords[-1, 1], marker="*", s=160,
               color="#d95f02", label="End")
    ax.set_xlabel("MDS dim 1")
    ax.set_ylabel("MDS dim 2")
    ax.set_title("Trajectory of best individual (MDS)")
    ax.legend()
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Generation")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_population_weight_stats(best_history: Sequence[np.ndarray],
                                 population_history: Sequence[np.ndarray],
                                 save_path: str,
                                 top_variables: int = 4) -> None:
    """Line charts showing best weights and population statistics."""
    best_matrix = _as_matrix(best_history)
    if len(population_history) != len(best_matrix):
        raise ValueError("Population history length must match best history.")

    pop_mean = []
    pop_std = []
    for pop in population_history:
        pop_mean.append(np.mean(pop, axis=0))
        pop_std.append(np.std(pop, axis=0))
    pop_mean = np.vstack(pop_mean)
    pop_std = np.vstack(pop_std)

    variances = best_matrix.var(axis=0)
    idx = np.argsort(variances)[-min(top_variables, best_matrix.shape[1]):]
    idx.sort()

    generations = np.arange(best_matrix.shape[0])
    n_vars = len(idx)
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]

    for ax, var_idx in zip(axes, idx):
        ax.plot(generations, pop_mean[:, var_idx], color="#1f77b4", label="Population mean")
        ax.fill_between(generations,
                        pop_mean[:, var_idx] - pop_std[:, var_idx],
                        pop_mean[:, var_idx] + pop_std[:, var_idx],
                        alpha=0.2, color="#1f77b4", label="Population ±1 std")
        ax.plot(generations, best_matrix[:, var_idx], color="#d62728",
                linewidth=2.0, label="Best individual")
        ax.set_ylabel(f"w[{var_idx}]")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Generation")
    fig.suptitle("Population statistics per variable", fontsize=14)
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
