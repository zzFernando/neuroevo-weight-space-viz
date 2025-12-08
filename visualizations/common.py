"""
Common helpers for visualization styling, color normalization, and scatter utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors


# -----------------------------------------------------------------------------
# Color normalization and colormaps
# -----------------------------------------------------------------------------
class EqualizedHistNorm(mcolors.Normalize):
    def __init__(self, values: np.ndarray, bins: int = 256):
        values = np.asarray(values)
        if values.size == 0:
            self.bin_edges = np.array([0.0, 1.0])
            self.cdf = np.array([0.0, 1.0])
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = np.percentile(values, [1, 99])
            if vmax <= vmin:
                vmax = vmin + 1e-8
            clipped = np.clip(values, vmin, vmax)
            hist, bin_edges = np.histogram(clipped, bins=bins, range=(vmin, vmax), density=True)
            cdf = np.cumsum(hist)
            cdf = cdf / cdf[-1]
            self.bin_edges = bin_edges
            self.cdf = cdf
        super().__init__(vmin=vmin, vmax=vmax)

    def __call__(self, value, clip=None):
        arr = np.ma.array(value, copy=True)
        arr = np.ma.clip(arr, self.vmin, self.vmax)
        idx = np.searchsorted(self.bin_edges[1:], arr.filled(self.vmin), side="right")
        idx = np.clip(idx, 0, len(self.cdf) - 1)
        out = self.cdf[idx]
        return np.ma.array(out, mask=np.ma.getmask(arr))


class AdaptiveColorNorm:
    def __init__(self, values: np.ndarray):
        self.values = np.asarray(values)

    def power(self, gamma: float = 0.3):
        lo, hi = np.percentile(self.values, [1, 99]) if self.values.size else (0, 1)
        hi = max(hi, lo + 1e-8)
        return mcolors.PowerNorm(gamma=gamma, vmin=lo, vmax=hi)

    def clipped(self, low: float = 1.0, high: float = 99.0):
        if self.values.size:
            vmin, vmax = np.percentile(self.values, [low, high])
            vmax = max(vmax, vmin + 1e-8)
        else:
            vmin, vmax = 0.0, 1.0
        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    def equalized(self, bins: int = 256):
        return EqualizedHistNorm(self.values, bins=bins)


def get_discrete_cmap(name: str, n: int = 20):
    custom = {
        "fitness_map": ["#0d0887", "#7e03a8", "#cb4679", "#f89441", "#f0f921"],
    }
    if name == "fitness_map":
        return mcolors.LinearSegmentedColormap.from_list("fitness_map", custom["fitness_map"], N=n)
    return plt.cm.get_cmap(name, n)


# -----------------------------------------------------------------------------
# Quantile-based fitness colormap
# -----------------------------------------------------------------------------
def build_fitness_quantile_colormap(fitness_values: np.ndarray, n_bins: int = 9, cmap_name: str = "plasma"):
    """
    Build a quantile-based discrete colormap for fitness values.
    """
    fitness_values = np.asarray(fitness_values)
    finite_mask = np.isfinite(fitness_values)
    vals = fitness_values[finite_mask]

    if vals.size == 0:
        boundaries = np.linspace(0, 1, n_bins + 1)
    else:
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        boundaries = np.quantile(vals, quantiles)
    boundaries[0] -= 1e-9
    boundaries[-1] += 1e-9

    base_cmap = plt.get_cmap(cmap_name)
    cmap = mcolors.ListedColormap(base_cmap(np.linspace(0, 1, n_bins)))
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    return cmap, norm, boundaries


# -----------------------------------------------------------------------------
# Scatter helpers
# -----------------------------------------------------------------------------
def density_based_alpha(points: np.ndarray, base_alpha: float = 0.4, k: int = 15) -> np.ndarray:
    """
    Compute alphas inversely proportional to local density using k-NN distances.
    """
    if len(points) == 0:
        return np.array([])
    nbrs = NearestNeighbors(n_neighbors=min(k, len(points))).fit(points)
    dist, _ = nbrs.kneighbors(points)
    mean_dist = dist[:, 1:].mean(axis=1)  # skip self
    norm = (mean_dist - mean_dist.min()) / (np.ptp(mean_dist) + 1e-8)
    alpha = base_alpha * (0.4 + 0.6 * norm)
    return alpha


def scatter_2d(
    ax: plt.Axes,
    pts: np.ndarray,
    color_values: np.ndarray,
    cmap,
    norm,
    size: float = 10.0,
    base_alpha: float = 0.55,
    density_alpha: bool = True,
    grid: bool = True,
) -> plt.Collection:
    if len(pts) == 0:
        return ax.scatter([], [])
    alpha = density_based_alpha(pts, base_alpha) if density_alpha else np.full(len(pts), base_alpha)
    sc = ax.scatter(
        pts[:, 0],
        pts[:, 1],
        c=color_values,
        cmap=cmap,
        norm=norm,
        s=size,
        alpha=alpha,
    )
    if grid:
        ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    return sc
