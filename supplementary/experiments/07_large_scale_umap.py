from __future__ import annotations

"""
Exp 7: Large-scale UMAP visualization (1M+ points).

Strategy
--------
- Fit UMAP on a stratified subsample (FIT_N points).
- .transform() the remaining points in BATCH_SIZE chunks.
- Render with np.histogram2d + imshow — O(grid²) regardless of n.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import umap

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.shared import CACHE_DIR, FIGURES_DIR, ensure_dirs

SEED       = 42
N_TOTAL    = 1_000_000
N_DIM      = 50
FIT_N      = 50_000   # subsample used to fit UMAP
BATCH_SIZE = 100_000  # transform batch to avoid OOM
GRID_RES   = 800      # histogram resolution (pixels)

C0 = np.array([0.22, 0.49, 0.72])   # blue — class 0
C1 = np.array([0.89, 0.10, 0.11])   # red  — class 1


# ── Data generation ───────────────────────────────────────────────────────────

def make_binary_large(n: int = N_TOTAL, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """
    Binary problem in 50D with multi-modal class structure.

    Class 0 (50%):  3 tight clusters + 1 diffuse cloud
    Class 1 (50%):  3 tight clusters (partially overlapping C0) + 1 diffuse cloud

    Uses chunked generation to keep peak RAM low.
    """
    rng = np.random.default_rng(seed)
    half = n // 2

    # Base directions
    dirs = rng.standard_normal((8, N_DIM))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    def blob(direction, scale, radius, count):
        return direction * scale + rng.standard_normal((count, N_DIM)) * radius

    # Class 0: 3 tight + 1 diffuse
    n0a = int(half * 0.35)
    n0b = int(half * 0.30)
    n0c = int(half * 0.20)
    n0d = half - n0a - n0b - n0c

    X0 = np.vstack([
        blob( dirs[0],  6.0, 0.5, n0a),
        blob( dirs[1],  5.0, 0.6, n0b),
        blob( dirs[2],  4.0, 0.7, n0c),
        blob( dirs[3],  0.0, 3.0, n0d),   # diffuse cloud near origin
    ])

    # Class 1: 3 tight (shifted) + 1 diffuse (overlaps C0 cloud)
    n1a = int(half * 0.35)
    n1b = int(half * 0.30)
    n1c = int(half * 0.20)
    n1d = half - n1a - n1b - n1c

    X1 = np.vstack([
        blob(-dirs[0],  6.0, 0.5, n1a),
        blob(-dirs[1],  5.0, 0.6, n1b),
        blob( dirs[4],  4.5, 0.6, n1c),
        blob( dirs[3],  0.5, 3.5, n1d),   # overlaps with C0 diffuse region
    ])

    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.array([0]*len(X0) + [1]*len(X1), dtype=np.int8)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ── UMAP fit + transform ──────────────────────────────────────────────────────

def _stratified_sample(X, y, n, rng):
    idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
    half = n // 2
    s0 = rng.choice(idx0, min(half, len(idx0)), replace=False)
    s1 = rng.choice(idx1, min(half, len(idx1)), replace=False)
    idx = np.concatenate([s0, s1])
    rng.shuffle(idx)
    return idx


def fit_and_transform(X, y, n_neighbors: int = 15, min_dist: float = 0.1):
    rng = np.random.default_rng(SEED)

    # Stratified subsample for fitting
    fit_idx = _stratified_sample(X, y, FIT_N, rng)
    X_fit   = X[fit_idx]
    mask_fit = np.zeros(len(X), dtype=bool); mask_fit[fit_idx] = True

    print(f"  Fitting UMAP on {len(X_fit):,} points (n_neighbors={n_neighbors}, min_dist={min_dist})…")
    t0 = time.time()
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="euclidean",
        low_memory=True,
        random_state=SEED,
        verbose=False,
    )
    emb_fit = reducer.fit_transform(X_fit)
    print(f"  Fit done in {time.time()-t0:.1f}s")

    # Transform the rest in batches
    rest_idx = np.where(~mask_fit)[0]
    emb_rest = np.empty((len(rest_idx), 2), dtype=np.float32)
    print(f"  Transforming {len(rest_idx):,} remaining points in batches…")
    t0 = time.time()
    for start in range(0, len(rest_idx), BATCH_SIZE):
        chunk = rest_idx[start:start + BATCH_SIZE]
        emb_rest[start:start + len(chunk)] = reducer.transform(X[chunk])
        pct = min(start + BATCH_SIZE, len(rest_idx)) / len(rest_idx) * 100
        print(f"    {pct:.0f}%", end="\r", flush=True)
    print(f"  Transform done in {time.time()-t0:.1f}s")

    # Reassemble in original order
    emb = np.empty((len(X), 2), dtype=np.float32)
    emb[fit_idx]  = emb_fit
    emb[rest_idx] = emb_rest
    return emb


# ── Density rendering ─────────────────────────────────────────────────────────

def _hist2d(emb, y, cls, xlim, ylim, res):
    mask = y == cls
    h, _, _ = np.histogram2d(
        emb[mask, 0], emb[mask, 1],
        bins=res,
        range=[xlim, ylim],
    )
    return h.T  # (row=y, col=x)


def render_density(emb, y, out_path: Path, n_neighbors: int, min_dist: float) -> None:
    pad = 0.05
    x0, x1 = emb[:, 0].min(), emb[:, 0].max()
    y0, y1 = emb[:, 1].min(), emb[:, 1].max()
    xr = x1 - x0; yr = y1 - y0
    xlim = [x0 - pad*xr, x1 + pad*xr]
    ylim = [y0 - pad*yr, y1 + pad*yr]

    print("  Building 2D histograms…")
    h0 = _hist2d(emb, y, 0, xlim, ylim, GRID_RES).astype(float)
    h1 = _hist2d(emb, y, 1, xlim, ylim, GRID_RES).astype(float)

    # Log-scale density then normalise to [0,1]
    def norm(h):
        h = np.log1p(h)
        return h / max(h.max(), 1e-9)

    h0n = norm(h0)
    h1n = norm(h1)
    total = np.log1p(h0 + h1)
    total = total / max(total.max(), 1e-9)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

    # ── Panel 1: class 0 density ──────────────────────────────────────────
    ax = axes[0]
    ax.imshow(h0n, origin="lower", extent=[*xlim, *ylim],
              cmap="Blues", aspect="auto", interpolation="bilinear")
    ax.set_title("Class 0 density", fontsize=11)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    # ── Panel 2: class 1 density ──────────────────────────────────────────
    ax = axes[1]
    ax.imshow(h1n, origin="lower", extent=[*xlim, *ylim],
              cmap="Reds", aspect="auto", interpolation="bilinear")
    ax.set_title("Class 1 density", fontsize=11)
    ax.set_xlabel("UMAP-1")

    # ── Panel 3: blended class map ────────────────────────────────────────
    ax = axes[2]

    # RGB image: blend blue (C0) and red (C1) by relative density
    alpha = total[..., np.newaxis]          # overall density → brightness
    class1_ratio = np.where(               # 0 = all C0, 1 = all C1
        (h0 + h1) > 0,
        h1 / (h0 + h1 + 1e-9),
        0.5,
    )[..., np.newaxis]
    rgb = (1 - class1_ratio) * C0 + class1_ratio * C1  # interpolate colors
    # Darken low-density pixels → white background
    img = 1.0 - alpha * (1.0 - rgb)
    img = np.clip(img, 0, 1)

    ax.imshow(img, origin="lower", extent=[*xlim, *ylim],
              aspect="auto", interpolation="bilinear")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=C0, label="class 0"),
        Patch(color=C1, label="class 1"),
    ], fontsize=9, loc="upper left")
    ax.set_title("Class map (blended)", fontsize=11)
    ax.set_xlabel("UMAP-1")

    fig.suptitle(
        f"Large-scale UMAP  |  n={len(y):,}  dim={N_DIM}  "
        f"n_neighbors={n_neighbors}  min_dist={min_dist}  seed={SEED}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _emb_cache_path(n_neighbors, min_dist):
    return CACHE_DIR / f"exp7_emb_nn{n_neighbors}_md{min_dist}_n{N_TOTAL}.npy"

def _data_cache_path():
    return CACHE_DIR / f"exp7_data_n{N_TOTAL}.npz"


def main(n_neighbors: int, min_dist: float, force_rerun: bool) -> None:
    ensure_dirs()

    # Data
    dc = _data_cache_path()
    if dc.exists() and not force_rerun:
        print("Loading cached data…")
        d = np.load(dc)
        X, y = d["X"], d["y"]
    else:
        print(f"Generating {N_TOTAL:,} points in {N_DIM}D…")
        t0 = time.time()
        X, y = make_binary_large()
        print(f"  Generated in {time.time()-t0:.1f}s  RAM ≈ {X.nbytes/1e9:.2f} GB")
        np.savez_compressed(dc, X=X, y=y)
        print(f"  Data cached to {dc}")

    print(f"Dataset: {X.shape}  class 0={( y==0).sum():,}  class 1={(y==1).sum():,}")

    # Embedding
    ec = _emb_cache_path(n_neighbors, min_dist)
    if ec.exists() and not force_rerun:
        print("Loading cached embedding…")
        emb = np.load(ec)
    else:
        emb = fit_and_transform(X, y, n_neighbors=n_neighbors, min_dist=min_dist)
        np.save(ec, emb)
        print(f"  Embedding cached to {ec}")

    # Render
    out = FIGURES_DIR / f"exp7_large_umap_nn{n_neighbors}_md{min_dist}.png"
    print("Rendering density maps…")
    render_density(emb, y, out, n_neighbors=n_neighbors, min_dist=min_dist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 7: large-scale UMAP (1M points)")
    parser.add_argument("--n-neighbors", type=int,   default=15)
    parser.add_argument("--min-dist",    type=float, default=0.1)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()
    main(n_neighbors=args.n_neighbors, min_dist=args.min_dist, force_rerun=args.force_rerun)
