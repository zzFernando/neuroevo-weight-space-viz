"""
visualize_datashader.py — Datashader-based visualization for neuroevolution runs.

Pipeline
--------
1. Load .npz produced by neuroevolve_brax.py
   (populations: G×P×D, fitnesses: G×P)
2. Joint UMAP on all G·P individuals → Z_flat (G·P, 2)
   Optional PCA pre-reduction if D > 5000.
3. Reference-anchored alignment (Eq. 1 of the paper):
   Z_aligned[k] = (1 − λ) · Z[k] + λ · Z[0]
4. Build a pandas DataFrame: x, y, generation, fitness
5. Render four Datashader figures per seed:
   (a) population density (log scale, inferno)
   (b) colored by generation (viridis)
   (c) colored by fitness   (plasma)
   (d) density + matplotlib streamplot velocity field

Embedding cached to <stem>_embedding.npz — re-runs skip UMAP.

Usage
-----
Single seed:
    python visualize_datashader.py runs/seed42.npz --out_dir figures/datashader/

Multi-seed composite:
    python visualize_datashader.py runs/stress_seed*.npz \\
        --out_dir figures/datashader/ --multi_seed_grid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import colorcet as cc
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from sklearn.decomposition import PCA
import umap
from paths import FIGURES_DIR

matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 10})

PCA_THRESHOLD = 5_000   # apply PCA if D > this
PCA_COMPONENTS = 50


# ── Data loading ──────────────────────────────────────────────────────────────

def load_run(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "populations": data["populations"],   # (G, P, D)
        "fitnesses":   data["fitnesses"],      # (G, P)
        "meta":        data["meta"].item() if "meta" in data.files else {},
    }


# ── Embedding ─────────────────────────────────────────────────────────────────

def compute_embedding(
    populations: np.ndarray,
    subsample: Optional[int] = None,
    random_state: int = 0,
) -> np.ndarray:
    """Joint UMAP over all G·P individuals.  Returns Z_flat (G·P, 2)."""
    G, P, D = populations.shape
    flat = populations.reshape(G * P, D).astype(np.float32)

    if D > PCA_THRESHOLD:
        print(f"  PCA {D}→{PCA_COMPONENTS} before UMAP…")
        flat = PCA(n_components=PCA_COMPONENTS, random_state=random_state).fit_transform(flat)

    if subsample and subsample < len(flat):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(flat), subsample, replace=False)
        print(f"  Fitting UMAP on {subsample:,}/{len(flat):,} (subsample)…")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                            random_state=random_state)
        reducer.fit(flat[idx])
        print(f"  Transforming full {len(flat):,} points…")
        Z_flat = reducer.transform(flat)
    else:
        print(f"  Fitting UMAP on {len(flat):,} points…")
        Z_flat = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                           random_state=random_state).fit_transform(flat)

    return Z_flat.astype(np.float32)


def align_embedding(Z_flat: np.ndarray, G: int, P: int, lambda_align: float) -> np.ndarray:
    """Eq. 1: Z_aligned[k] = (1−λ)·Z[k] + λ·Z[0].  Returns (G, P, 2)."""
    Z = Z_flat.reshape(G, P, 2)
    Z_ref = Z[0]
    return ((1 - lambda_align) * Z + lambda_align * Z_ref[None]).astype(np.float32)


def get_or_compute_embedding(
    npz_path: Path,
    lambda_align: float,
    subsample: Optional[int],
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load cached embedding or run the full pipeline.

    Returns Z_aligned (G,P,2), fitnesses_flat (G·P,), generations_flat (G·P,), meta.
    """
    cache = npz_path.with_name(npz_path.stem + "_embedding.npz")
    run   = load_run(npz_path)
    G, P  = run["fitnesses"].shape

    if cache.exists() and not force:
        print(f"  Loading cached embedding: {cache.name}")
        cached = np.load(cache)
        return (
            cached["Z_aligned"],
            cached["fitnesses_flat"],
            cached["generations_flat"],
            run["meta"],
        )

    Z_flat    = compute_embedding(run["populations"], subsample=subsample)
    Z_aligned = align_embedding(Z_flat, G, P, lambda_align)

    gens_flat = np.repeat(np.arange(G, dtype=np.int32), P)
    fits_flat = run["fitnesses"].reshape(-1).astype(np.float32)

    np.savez_compressed(cache,
                        Z_aligned=Z_aligned,
                        fitnesses_flat=fits_flat,
                        generations_flat=gens_flat)
    print(f"  Embedding cached → {cache.name}")
    return Z_aligned, fits_flat, gens_flat, run["meta"]


# ── Velocity field ────────────────────────────────────────────────────────────

def compute_velocity_field(Z_aligned: np.ndarray, n_bins: int = 22) -> tuple:
    """Grid-binned velocity field (Section 3.3 of the paper).

    Returns x_edges, y_edges, U, V (all (n_bins,*) arrays).
    """
    G, P, _ = Z_aligned.shape
    pts = Z_aligned.reshape(-1, 2)
    pad = 0.05 * (pts[:, 0].ptp())
    x_edges = np.linspace(pts[:, 0].min() - pad, pts[:, 0].max() + pad, n_bins + 1)
    y_edges = np.linspace(pts[:, 1].min() - pad, pts[:, 1].max() + pad, n_bins + 1)

    U = np.zeros((n_bins, n_bins), dtype=np.float32)
    V = np.zeros((n_bins, n_bins), dtype=np.float32)
    C = np.zeros((n_bins, n_bins), dtype=np.float32)

    for g in range(G - 1):
        z0, z1 = Z_aligned[g], Z_aligned[g + 1]
        mid = 0.5 * (z0 + z1)
        ix = np.clip(np.searchsorted(x_edges, mid[:, 0]) - 1, 0, n_bins - 1)
        iy = np.clip(np.searchsorted(y_edges, mid[:, 1]) - 1, 0, n_bins - 1)
        d  = z1 - z0
        np.add.at(U, (iy, ix), d[:, 0])
        np.add.at(V, (iy, ix), d[:, 1])
        np.add.at(C, (iy, ix), 1)

    mask = C > 0
    U[mask] /= C[mask]
    V[mask] /= C[mask]
    return x_edges, y_edges, uniform_filter(U, 3), uniform_filter(V, 3)


# ── Datashader helpers ────────────────────────────────────────────────────────

def _make_canvas(df: pd.DataFrame, width: int, height: int,
                 xr: tuple, yr: tuple) -> ds.Canvas:
    return ds.Canvas(plot_width=width, plot_height=height,
                     x_range=xr, y_range=yr)


def _ds_img_to_ax(img_ds, ax: plt.Axes, xr: tuple, yr: tuple) -> None:
    """Blit a datashader Image onto a matplotlib Axes with correct extent."""
    img_arr = np.array(img_ds.to_pil())   # (H, W, 4) uint8, row-0 = top
    ax.imshow(img_arr,
              extent=[xr[0], xr[1], yr[0], yr[1]],
              origin="upper", aspect="auto")
    ax.set_xlim(xr); ax.set_ylim(yr)


def _cmap_from_cc(cc_list) -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list("_cc", cc_list)


def _add_colorbar(ax, cmap, vmin, vmax, label):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label=label)


def _decorate(ax, title, xr, yr):
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(title, fontsize=9)


def _save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Four render functions ─────────────────────────────────────────────────────

def render_density(
    df: pd.DataFrame, xr: tuple, yr: tuple,
    width: int, height: int, out_path: Path, title: str,
) -> None:
    canvas = _make_canvas(df, width, height, xr, yr)
    agg    = canvas.points(df, "x", "y", ds.count())
    img    = tf.shade(agg, cmap=cc.fire, how="log")
    img    = tf.set_background(img, "white")

    fig, ax = plt.subplots(figsize=(7, 6))
    _ds_img_to_ax(img, ax, xr, yr)
    _decorate(ax, title, xr, yr)
    ax.text(0.02, 0.02, f"n = {len(df):,}", transform=ax.transAxes,
            fontsize=8, color="#888")
    _save_fig(fig, out_path)


def render_by_generation(
    df: pd.DataFrame, xr: tuple, yr: tuple,
    width: int, height: int, out_path: Path, title: str,
) -> None:
    canvas = _make_canvas(df, width, height, xr, yr)
    agg    = canvas.points(df, "x", "y", ds.mean("generation"))
    img    = tf.shade(agg, cmap=cc.bmy, how="linear")
    img    = tf.set_background(img, "white")

    fig, ax = plt.subplots(figsize=(7, 6))
    _ds_img_to_ax(img, ax, xr, yr)
    _decorate(ax, title, xr, yr)
    _add_colorbar(ax, _cmap_from_cc(cc.bmy),
                  df["generation"].min(), df["generation"].max(), "generation")
    _save_fig(fig, out_path)


def render_by_fitness(
    df: pd.DataFrame, xr: tuple, yr: tuple,
    width: int, height: int, out_path: Path, title: str,
) -> None:
    canvas = _make_canvas(df, width, height, xr, yr)
    agg    = canvas.points(df, "x", "y", ds.mean("fitness"))
    img    = tf.shade(agg, cmap=cc.CET_L19, how="linear")
    img    = tf.set_background(img, "white")

    fig, ax = plt.subplots(figsize=(7, 6))
    _ds_img_to_ax(img, ax, xr, yr)
    _decorate(ax, title, xr, yr)
    _add_colorbar(ax, _cmap_from_cc(cc.CET_L19),
                  df["fitness"].min(), df["fitness"].max(), "fitness (episode return)")
    _save_fig(fig, out_path)


def render_velocity_overlay(
    df: pd.DataFrame, Z_aligned: np.ndarray,
    xr: tuple, yr: tuple,
    width: int, height: int, out_path: Path, title: str,
) -> None:
    # Datashader density background
    canvas = _make_canvas(df, width, height, xr, yr)
    agg    = canvas.points(df, "x", "y", ds.count())
    img    = tf.shade(agg, cmap=cc.blues, how="log")
    img    = tf.set_background(img, "white")

    # Velocity field
    xe, ye, U, V = compute_velocity_field(Z_aligned, n_bins=22)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    Xc, Yc = np.meshgrid(xc, yc)
    speed  = np.sqrt(U ** 2 + V ** 2)

    fig, ax = plt.subplots(figsize=(7, 6))
    _ds_img_to_ax(img, ax, xr, yr)
    if speed.max() > 0:
        ax.streamplot(Xc, Yc, U, V,
                      color=speed, cmap="magma",
                      density=1.3, linewidth=0.9, arrowsize=0.9)
    _decorate(ax, title, xr, yr)
    _save_fig(fig, out_path)


# ── Single-seed pipeline ──────────────────────────────────────────────────────

def process_single(
    npz_path: Path,
    out_dir: Path,
    lambda_align: float,
    width: int,
    height: int,
    subsample: Optional[int],
    force_embed: bool,
) -> None:
    stem = npz_path.stem
    print(f"\n[{stem}]")

    Z_aligned, fits_flat, gens_flat, meta = get_or_compute_embedding(
        npz_path, lambda_align, subsample, force=force_embed
    )

    G, P, _ = Z_aligned.shape
    Z_flat  = Z_aligned.reshape(-1, 2)

    df = pd.DataFrame({
        "x":          Z_flat[:, 0].astype(float),
        "y":          Z_flat[:, 1].astype(float),
        "generation": gens_flat.astype(float),
        "fitness":    fits_flat.astype(float),
    })

    pad = 0.05
    x0, x1 = float(df.x.min()), float(df.x.max()); dx = x1 - x0
    y0, y1 = float(df.y.min()), float(df.y.max()); dy = y1 - y0
    xr = (x0 - pad*dx, x1 + pad*dx)
    yr = (y0 - pad*dy, y1 + pad*dy)

    env  = meta.get("env", "env")
    seed = meta.get("seed", "?")
    base_title = f"{env} · seed {seed} · λ={lambda_align} · n={len(df):,}"

    render_density(df, xr, yr, width, height,
                   out_dir / f"{stem}_density.png",
                   f"Population density\n{base_title}")

    render_by_generation(df, xr, yr, width, height,
                         out_dir / f"{stem}_by_gen.png",
                         f"By generation\n{base_title}")

    render_by_fitness(df, xr, yr, width, height,
                      out_dir / f"{stem}_by_fitness.png",
                      f"By fitness\n{base_title}")

    render_velocity_overlay(df, Z_aligned, xr, yr, width, height,
                            out_dir / f"{stem}_velocity_field.png",
                            f"Velocity field\n{base_title}")


# ── Multi-seed composite grid ─────────────────────────────────────────────────

def make_multi_seed_grid(
    paths: list[Path],
    out_dir: Path,
    lambda_align: float,
    subsample: Optional[int],
    force_embed: bool,
    panel_px: int = 600,
) -> None:
    """One row per seed, four columns: density | by gen | by fitness | velocity."""
    n_seeds = len(paths)
    col_labels = ["Density", "By generation", "By fitness", "Velocity field"]
    n_cols = 4

    fig, axes = plt.subplots(
        n_seeds, n_cols,
        figsize=(4.5 * n_cols, 3.8 * n_seeds),
        dpi=150,
    )
    if n_seeds == 1:
        axes = axes[np.newaxis, :]

    for row, npz_path in enumerate(paths):
        stem = npz_path.stem
        print(f"\n[multi-seed] {stem}")

        Z_aligned, fits_flat, gens_flat, meta = get_or_compute_embedding(
            npz_path, lambda_align, subsample, force=force_embed
        )

        G, P, _ = Z_aligned.shape
        Z_flat  = Z_aligned.reshape(-1, 2)

        df = pd.DataFrame({
            "x":          Z_flat[:, 0].astype(float),
            "y":          Z_flat[:, 1].astype(float),
            "generation": gens_flat.astype(float),
            "fitness":    fits_flat.astype(float),
        })

        pad = 0.05
        x0, x1 = float(df.x.min()), float(df.x.max()); dx = x1 - x0
        y0, y1 = float(df.y.min()), float(df.y.max()); dy = y1 - y0
        xr = (x0 - pad*dx, x1 + pad*dx)
        yr = (y0 - pad*dy, y1 + pad*dy)

        seed = meta.get("seed", "?")
        env  = meta.get("env", "env")

        canvas = ds.Canvas(plot_width=panel_px, plot_height=panel_px,
                           x_range=xr, y_range=yr)

        panels = [
            ("density",  tf.shade(canvas.points(df, "x", "y", ds.count()),
                                  cmap=cc.fire, how="log")),
            ("by_gen",   tf.shade(canvas.points(df, "x", "y", ds.mean("generation")),
                                  cmap=cc.bmy, how="linear")),
            ("by_fit",   tf.shade(canvas.points(df, "x", "y", ds.mean("fitness")),
                                  cmap=cc.CET_L19, how="linear")),
        ]

        for col, (_, img_ds) in enumerate(panels):
            ax = axes[row, col]
            img_ds = tf.set_background(img_ds, "white")
            _ds_img_to_ax(img_ds, ax, xr, yr)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"seed {seed}\n({env})", fontsize=8)
            if row == 0:
                ax.set_title(col_labels[col], fontsize=9)

        # Velocity field panel
        ax = axes[row, 3]
        img_bg = tf.set_background(
            tf.shade(canvas.points(df, "x", "y", ds.count()), cmap=cc.blues, how="log"),
            "white",
        )
        _ds_img_to_ax(img_bg, ax, xr, yr)
        xe, ye, U, V = compute_velocity_field(Z_aligned, n_bins=22)
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])
        Xc, Yc = np.meshgrid(xc, yc)
        speed = np.sqrt(U ** 2 + V ** 2)
        if speed.max() > 0:
            ax.streamplot(Xc, Yc, U, V, color=speed, cmap="magma",
                          density=1.2, linewidth=0.8, arrowsize=0.8)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(col_labels[3], fontsize=9)

    fig.suptitle(
        f"Multi-seed summary · λ={lambda_align} · {n_seeds} seeds",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    out = out_dir / "multi_seed_summary.png"
    _save_fig(fig, out, dpi=150)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Datashader visualization for neuroevolution .npz files"
    )
    parser.add_argument("inputs", nargs="+", type=Path,
                        help=".npz file(s) produced by neuroevolve_brax.py")
    parser.add_argument("--lambda_align", type=float, default=0.8,
                        help="Alignment strength (default 0.8)")
    parser.add_argument("--width",   type=int, default=1600)
    parser.add_argument("--height",  type=int, default=1600)
    parser.add_argument("--out_dir", type=Path, default=FIGURES_DIR / "datashader")
    parser.add_argument("--multi_seed_grid", action="store_true",
                        help="Produce a single composite figure for all inputs")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Fit UMAP on N random points, transform the rest")
    parser.add_argument("--force_embed", action="store_true",
                        help="Ignore cached _embedding.npz and recompute")
    args = parser.parse_args()

    paths = sorted(args.inputs)
    for p in paths:
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr); sys.exit(1)

    if args.multi_seed_grid:
        make_multi_seed_grid(
            paths, args.out_dir, args.lambda_align,
            args.subsample, args.force_embed,
        )
    else:
        for p in paths:
            process_single(
                p, args.out_dir, args.lambda_align,
                args.width, args.height, args.subsample, args.force_embed,
            )


if __name__ == "__main__":
    main()
