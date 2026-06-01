from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.shared import FIGURES_DIR, ensure_dirs

SEED = 42
N_DIM = 50

N_NEIGHBORS_GRID = [5, 15, 30, 50]
MIN_DIST_GRID    = [0.0, 0.1, 0.3, 0.5]

# Colors for binary classes
C0 = "#377EB8"  # blue  — class 0
C1 = "#E41A1C"  # red   — class 1


# ── Synthetic binary dataset ──────────────────────────────────────────────────

def make_binary(seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """
    1500-point binary dataset in 50D with realistic structure:

    Class 0 (750 pts):
      - 1 large tight cluster       (300 pts)
      - 1 smaller tight cluster     (200 pts)
      - 1 diffuse cloud             (250 pts)

    Class 1 (750 pts):
      - 1 large tight cluster       (300 pts, well-separated from C0)
      - 1 smaller tight cluster     (200 pts, near C0 diffuse cloud → overlap)
      - 1 diffuse cloud             (250 pts)

    This gives: easy regions, hard overlap regions, and diffuse noise — a realistic
    binary problem to stress-test UMAP parameter choices.
    """
    rng = np.random.default_rng(seed)

    def cluster(center_vec, radius, n):
        c = np.array(center_vec)
        c = c / np.linalg.norm(c)
        return c * 5 + rng.standard_normal((n, N_DIM)) * radius

    # Orthogonal base directions
    d0 = rng.standard_normal(N_DIM); d0 /= np.linalg.norm(d0)
    d1 = rng.standard_normal(N_DIM); d1 -= d1.dot(d0) * d0; d1 /= np.linalg.norm(d1)
    d2 = rng.standard_normal(N_DIM); d2 -= d2.dot(d0)*d0 + d2.dot(d1)*d1; d2 /= np.linalg.norm(d2)

    parts0 = [
        cluster( d0,          0.4, 300),   # large tight — clearly class 0
        cluster( d0 + d2,     0.5, 200),   # smaller tight cluster
        cluster( d1,          2.0, 250),   # diffuse cloud, partial overlap
    ]
    parts1 = [
        cluster(-d0,          0.4, 300),   # large tight — clearly class 1
        cluster( d1 + 0.3*d0, 0.6, 200),  # overlaps with C0 diffuse region
        cluster(-d1,          2.0, 250),   # diffuse cloud, partial overlap
    ]

    X0 = np.vstack(parts0)
    X1 = np.vstack(parts1)
    X  = np.vstack([X0, X1])
    y  = np.array([0]*len(X0) + [1]*len(X1), dtype=int)

    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ── UMAP ─────────────────────────────────────────────────────────────────────

def embed(X: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    return umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="euclidean",
        random_state=SEED,
    ).fit_transform(X)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _scatter_binary(
    ax: plt.Axes,
    emb: np.ndarray,
    y: np.ndarray,
    s: float = 7,
    alpha: float = 0.55,
    legend: bool = False,
) -> None:
    ax.set_facecolor("white")
    for cls, color, label in [(0, C0, "class 0"), (1, C1, "class 1")]:
        mask = y == cls
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=color, label=label,
            s=s, alpha=alpha, edgecolors="none",
            rasterized=True,
        )
    if legend:
        ax.legend(fontsize=9, markerscale=2, loc="upper left", framealpha=0.85)
    ax.set_xticks([]); ax.set_yticks([])


def plot_param_grid(X: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    n_rows, n_cols = len(N_NEIGHBORS_GRID), len(MIN_DIST_GRID)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.8 * n_rows),
        dpi=130,
    )

    for r, nn in enumerate(tqdm(N_NEIGHBORS_GRID, desc="n_neighbors")):
        for c, md in enumerate(MIN_DIST_GRID):
            ax  = axes[r, c]
            emb = embed(X, n_neighbors=nn, min_dist=md)
            _scatter_binary(ax, emb, y, legend=(r == 0 and c == 0))
            if r == 0:
                ax.set_title(f"min_dist={md}", fontsize=11)
            if c == 0:
                ax.set_ylabel(f"n_neighbors={nn}", fontsize=11)

    fig.suptitle(
        f"UMAP param grid — binary problem  |  n={len(X)}  dim={X.shape[1]}  seed={SEED}",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_overview(
    X: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> None:
    emb = embed(X, n_neighbors=n_neighbors, min_dist=min_dist)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)

    _scatter_binary(axes[0], emb, y, s=12, alpha=0.65, legend=True)
    axes[0].set_xlabel("UMAP-1"); axes[0].set_ylabel("UMAP-2")
    axes[0].set_title(f"n_neighbors={n_neighbors}  min_dist={min_dist}")

    ax = axes[1]
    ax.set_facecolor("white")
    ax.hist(emb[y == 0, 0], bins=40, color=C0, alpha=0.6, label="class 0", density=True)
    ax.hist(emb[y == 1, 0], bins=40, color=C1, alpha=0.6, label="class 1", density=True)
    ax.set_xlabel("UMAP-1 projection"); ax.set_ylabel("density")
    ax.set_title("Class separation along UMAP-1")
    ax.legend(fontsize=9)

    fig.suptitle(
        f"Binary UMAP  |  n={len(X)}  dim={X.shape[1]}  seed={SEED}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(param_grid: bool, n_neighbors: int, min_dist: float) -> None:
    ensure_dirs()

    print(f"Generating binary dataset: n={1500}, dim={N_DIM}, seed={SEED}…")
    X, y = make_binary()
    print(f"  class 0: {(y==0).sum()}  class 1: {(y==1).sum()}")

    if param_grid:
        out = FIGURES_DIR / "exp6_binary_param_grid.png"
        plot_param_grid(X, y, out)
    else:
        out = FIGURES_DIR / f"exp6_binary_nn{n_neighbors}_md{min_dist}.png"
        plot_overview(X, y, out, n_neighbors=n_neighbors, min_dist=min_dist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 6: UMAP param demo — binary problem")
    parser.add_argument("--param-grid",  action="store_true")
    parser.add_argument("--n-neighbors", type=int,   default=15)
    parser.add_argument("--min-dist",    type=float, default=0.1)
    args = parser.parse_args()
    main(param_grid=args.param_grid, n_neighbors=args.n_neighbors, min_dist=args.min_dist)
