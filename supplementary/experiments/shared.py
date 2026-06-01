from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage
from scipy.ndimage import convolve
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from visualizations.vector_field import _compute_velocity_grid

_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _ROOT / "results"
FIGURES_DIR = _ROOT / "figures"
CACHE_DIR = _ROOT / "cache"


def set_science_style() -> None:
    """Apply the SciencePlots publication style (no-latex: no system LaTeX needed).

    Idempotent and safe to call at import time of an experiment. Keeps a couple of
    overrides so dense multi-panel scatter grids stay legible.
    """
    import matplotlib as mpl
    import scienceplots  # noqa: F401  (registers the styles)

    plt.style.use(["science", "no-latex", "grid"])
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.titlesize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
    })


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def count_attractors_dbscan(
    embedding: np.ndarray,
    eps: float | None = None,
    min_samples: int = 5,
) -> int:
    """Count macro-clusters using DBSCAN with percentile-based adaptive eps.

    Uses the 90th percentile of k-NN distances as eps, which captures
    macro-cluster scale while ignoring local noise. Scale-invariant across
    methods with very different spreads (PCA, t-SNE, UMAP).
    """
    if embedding.size == 0 or len(embedding) < min_samples:
        return 0
    if eps is not None:
        effective_eps = eps
    else:
        k = min(min_samples, len(embedding) - 1)
        nn = NearestNeighbors(n_neighbors=k).fit(embedding)
        distances, _ = nn.kneighbors(embedding)
        # 90th percentile of distance to k-th neighbor: macro-cluster scale
        effective_eps = float(np.percentile(distances[:, -1], 90))
    labels = DBSCAN(eps=effective_eps, min_samples=min_samples).fit_predict(embedding)
    return int(len(set(labels)) - (1 if -1 in labels else 0))


def temporal_coherence(embeddings: Sequence[np.ndarray]) -> float:
    if len(embeddings) < 2:
        return float("nan")

    corrs = []
    for a, b in zip(embeddings, embeddings[1:]):
        if a.shape != b.shape or a.size == 0:
            corrs.append(np.nan)
            continue
        x = np.array(a.ravel(), dtype=float)
        y = np.array(b.ravel(), dtype=float)
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            corrs.append(0.0)
            continue
        # abs: UMAP can arbitrarily flip/rotate embeddings between independent runs,
        # producing negative correlation even for temporally coherent data
        corr = abs(np.corrcoef(x, y)[0, 1])
        corrs.append(float(corr))
    corrs = [c for c in corrs if not np.isnan(c)]
    return float(np.mean(corrs)) if corrs else float("nan")


def _smooth2d(arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if sigma <= 0:
        return arr
    kernel = np.ones((3, 3), dtype=float) / 9.0
    return convolve(arr, kernel, mode="nearest")


def plot_compact_vector_field(
    ax: plt.Axes,
    per_gen_embeddings: Sequence[np.ndarray],
    emb_all: np.ndarray,
    grid_res: int = 22,
    min_vectors_per_cell: int = 1,
) -> None:
    ax.set_facecolor("white")
    ax.scatter(
        emb_all[:, 0], emb_all[:, 1],
        s=2,
        alpha=0.12,
        color="#777777",
        edgecolors="none",
        zorder=1,
    )

    try:
        Xc, Yc, U, V, speed = _compute_velocity_grid(
            per_gen_embeddings, grid_res=grid_res, min_vectors_per_cell=min_vectors_per_cell
        )
    except Exception:
        ax.text(0.5, 0.5, "Vector field\nnot available", ha="center", va="center", fontsize=9, color="#666")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    U_f = np.ma.filled(U, 0.0).astype(float)
    V_f = np.ma.filled(V, 0.0).astype(float)
    U_f = _smooth2d(U_f, sigma=1.0)
    V_f = _smooth2d(V_f, sigma=1.0)
    speed_s = np.sqrt(U_f ** 2 + V_f ** 2)

    if np.any(speed_s > 0):
        ax.streamplot(
            Xc,
            Yc,
            U_f,
            V_f,
            color=speed_s,
            cmap=plt.cm.plasma,
            density=1.2,
            linewidth=1.0,
            arrowsize=1.0,
            zorder=2,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")


def plot_fitness_panel(
    ax: plt.Axes,
    embedding: np.ndarray,
    fitness: np.ndarray,
    title: str | None = None,
) -> plt.cm.ScalarMappable:
    fitness = np.array(fitness, dtype=float)
    fmin = float(np.percentile(fitness, 50))
    fmax = float(np.percentile(fitness, 99))
    if fmax == fmin:
        fmax = fmin + 1e-9
    fitness_pct = np.clip((fitness - fmin) / (fmax - fmin) * 100, 0, 100)
    norm = plt.Normalize(vmin=0, vmax=100)

    fnorm = (fitness - fitness.min()) / max(fitness.max() - fitness.min(), 1e-9)
    sizes = 3 + 14 * fnorm

    ax.set_facecolor("white")
    sc = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=fitness_pct,
        cmap="viridis",
        norm=norm,
        s=sizes,
        edgecolors="white",
        linewidths=0.25,
        alpha=0.85,
        rasterized=True,
        zorder=2,
    )

    best_idx = int(np.argmax(fitness))
    ax.scatter(
        embedding[best_idx, 0], embedding[best_idx, 1],
        marker="*", s=140,
        c="#D62728",
        edgecolors="white", linewidths=0.8,
        zorder=5,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return sc


def save_results_csv(metrics: Iterable[dict[str, Any]], path: Path | str) -> None:
    df = pd.DataFrame.from_records(list(metrics))
    df.to_csv(Path(path), index=False)
