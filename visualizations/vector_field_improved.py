"""
Improved vector field visualization — publication-ready 2×2 layout.

Layout
------
┌────────────────────────┬────────────────────────┐
│  A  Hexbin + Trail     │  B  Streamlines        │
│     (density + top5%)  │     + attractor marks  │
├────────────────────────┼────────────────────────┤
│  C  Density (KDE)      │  D  Divergence         │
│     contourf + basin   │     + annotated regions│
└────────────────────────┴────────────────────────┘

Design decisions
----------------
A  hexbin replaces raw scatter → reveals density without overplotting 16k pts.
   Top-5% fitness overlaid as ★ so high-fitness loci are never hidden.
   Last-5-gen trail (red fade-in) gives temporal sense in a static figure.

B  Local velocity minima detected via minimum_filter → annotated as attractors.
   Faint KDE contour background keeps context without competing with streamlines.

C  contourf (not imshow) → smooth isolines, white background = scientific look.
   Max-density cell annotated as "Basin" with white star + label box.

D  contourf with TwoSlopeNorm, zero boundary as solid black line (not dashed).
   Arrow annotations point to the peak exploration and exploitation cells.

Global
------
- Unified xlim/ylim from auto-crop across all panels
- Shared axes: no duplicate xlabels (top row) / ylabels (right col)
- Monospace info-box (dataset, algorithm, best fitness) in top-right
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, minimum_filter
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors

from utils import compute_aligned_umap_embedding
from visualizations.common import (
    build_fitness_quantile_colormap,
    get_discrete_cmap,
)
from visualizations.vector_field import _compute_velocity_grid


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def auto_crop_to_data(
    points: np.ndarray,
    padding: float = 0.08,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Tight axis limits around actual data, plus a fractional padding.

    Parameters
    ----------
    points  : (N, 2) embedding coordinates.
    padding : fraction of data range added symmetrically on each side.

    Returns
    -------
    xlim, ylim : each a (lo, hi) tuple.
    """
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    pad_x = padding * (x_max - x_min)
    pad_y = padding * (y_max - y_min)
    return (x_min - pad_x, x_max + pad_x), (y_min - pad_y, y_max + pad_y)


def adaptive_subsample(
    Xc: np.ndarray,
    Yc: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    reference_points: np.ndarray,
    keep_sparse_frac: float = 0.90,
    drop_dense_frac: float = 0.30,
    k_neighbors: int = 8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Drop grid vectors proportional to local point-cloud density.

    Dense cells (many nearby population points) → keep fewer vectors.
    Sparse cells → keep nearly all vectors.

    Returns flat arrays (Xs, Ys, Us, Vs) of surviving grid points.
    """
    if isinstance(U, np.ma.MaskedArray):
        valid_flat = ~U.mask.ravel()
    else:
        valid_flat = ~np.isnan(np.ma.filled(U, np.nan)).ravel()

    grid_pts = np.column_stack([Xc.ravel(), Yc.ravel()])
    valid_grid = grid_pts[valid_flat]

    if len(reference_points) == 0 or len(valid_grid) == 0:
        mask = valid_flat.reshape(Xc.shape)
        return Xc[mask], Yc[mask], np.ma.filled(U, 0.0)[mask], np.ma.filled(V, 0.0)[mask]

    k = min(k_neighbors, len(reference_points))
    nbrs = NearestNeighbors(n_neighbors=k).fit(reference_points)
    dists, _ = nbrs.kneighbors(valid_grid)
    mean_dist = dists.mean(axis=1)

    d_min, d_max = mean_dist.min(), mean_dist.max()
    sparseness = (mean_dist - d_min) / (d_max - d_min + 1e-12)
    keep_prob = drop_dense_frac + (keep_sparse_frac - drop_dense_frac) * sparseness

    rng = np.random.default_rng(seed)
    survive = rng.random(len(valid_grid)) < keep_prob

    survive_full = np.zeros(len(grid_pts), dtype=bool)
    survive_full[valid_flat] = survive
    mask_2d = survive_full.reshape(Xc.shape)

    return (
        Xc[mask_2d],
        Yc[mask_2d],
        np.ma.filled(U, 0.0)[mask_2d],
        np.ma.filled(V, 0.0)[mask_2d],
    )


def compute_divergence(
    U: np.ndarray,
    V: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
) -> np.ndarray:
    """
    div(F) = ∂U/∂x + ∂V/∂y  using central differences (numpy.gradient).

    Positive → sources / expansion (exploration).
    Negative → sinks   / contraction (exploitation).
    """
    U_arr = np.ma.filled(U, 0.0).astype(float)
    V_arr = np.ma.filled(V, 0.0).astype(float)

    dU_dx = np.gradient(U_arr, dx, axis=1)
    dV_dy = np.gradient(V_arr, dy, axis=0)
    div = dU_dx + dV_dy

    if isinstance(U, np.ma.MaskedArray) or isinstance(V, np.ma.MaskedArray):
        mask_u = np.ma.getmaskarray(U)
        mask_v = np.ma.getmaskarray(V)
        return np.ma.array(div, mask=mask_u | mask_v)
    return div


def _kde_density_grid(
    points: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    grid_res: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Gaussian KDE on a regular grid. Returns (Xg, Yg, density)."""
    x_lin = np.linspace(xlim[0], xlim[1], grid_res)
    y_lin = np.linspace(ylim[0], ylim[1], grid_res)
    Xg, Yg = np.meshgrid(x_lin, y_lin)
    try:
        density = gaussian_kde(points.T, bw_method="scott")(
            np.vstack([Xg.ravel(), Yg.ravel()])
        ).reshape(Xg.shape)
    except (np.linalg.LinAlgError, ValueError):
        density = np.zeros_like(Xg)
    return Xg, Yg, density


def _find_attractors(
    speed: np.ndarray,
    Xc: np.ndarray,
    Yc: np.ndarray,
    n_attractors: int = 2,
    neighborhood: int = 5,
    speed_percentile: float = 25.0,
) -> np.ndarray:
    """
    Find up to `n_attractors` local velocity minima as candidate attractors.

    Returns array of shape (k, 2) with (x, y) coordinates in embedding space.
    """
    threshold = np.percentile(speed, speed_percentile)
    local_min_mask = (speed == minimum_filter(speed, size=neighborhood))
    candidate_mask = local_min_mask & (speed < threshold)
    indices = np.argwhere(candidate_mask)

    if len(indices) == 0:
        return np.empty((0, 2))

    # rank by lowest speed first
    scores = speed[indices[:, 0], indices[:, 1]]
    ranked = indices[np.argsort(scores)]
    top = ranked[:n_attractors]
    return np.array([[Xc[r, c], Yc[r, c]] for r, c in top])


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def plot_improved_vector_field(
    weights_by_gen: Sequence[np.ndarray],
    fitness_by_gen: Sequence[np.ndarray],
    # ── embedding ────────────────────────────────────────────────────────
    lambda_align: float = 0.8,
    random_state: int = 42,
    # ── velocity grid ────────────────────────────────────────────────────
    grid_res: int = 30,
    min_vectors_per_cell: int = 1,
    smoothing_sigma: float = 1.5,
    # ── crop & subsample ─────────────────────────────────────────────────
    crop_padding: float = 0.08,
    keep_sparse_frac: float = 0.90,
    drop_dense_frac: float = 0.30,
    # ── panel A ──────────────────────────────────────────────────────────
    hexbin_gridsize: int = 40,
    top_fitness_pct: float = 95.0,
    trail_gens: int = 5,
    # ── panel B ──────────────────────────────────────────────────────────
    streamline_density: float = 1.4,
    n_attractors: int = 2,
    # ── panel C ──────────────────────────────────────────────────────────
    density_grid_res: int = 100,
    density_contour_levels: int = 20,
    # ── fitness coloring ─────────────────────────────────────────────────
    fitness_bins: int = 24,
    cmap_gen: Optional[object] = None,
    cmap_fit: Optional[object] = None,
    # ── title & metadata ─────────────────────────────────────────────────
    title_params: Optional[Dict] = None,
    # ── figure ───────────────────────────────────────────────────────────
    figsize: Tuple[float, float] = (16, 10),
    dpi: int = 150,
    font_size: int = 10,
) -> plt.Figure:
    """
    Publication-ready 2×2 weight-space dynamics figure.

    Panel A — Hexbin + temporal trail + top-5% stars
        hexbin reveals density structure for thousands of overlapping points.
        Top `top_fitness_pct` percentile overlaid as ★ (never hidden).
        Last `trail_gens` generations drawn with opacity fade-in.

    Panel B — Streamlines + attractor markers
        scipy-smoothed velocity field → continuous flow lines coloured by speed.
        Local velocity minima detected and marked (candidate attractors / basins).

    Panel C — Time-weighted KDE density (contourf, white background)
        Later generations weight 5× heavier so the final attractor dominates.
        Max-density cell annotated as the convergence basin.

    Panel D — Divergence with arrow annotations
        div > 0 (red) = exploration, div < 0 (blue) = exploitation.
        Solid black zero-contour marks the transition boundary.
        Arrow annotations label the peak exploration/exploitation cells.

    Parameters
    ----------
    weights_by_gen  : weight matrices per generation, each (pop_size, n_weights).
    fitness_by_gen  : fitness scores per generation, each (pop_size,).
    lambda_align    : temporal alignment strength (0 = off, 1 = max).
    random_state    : UMAP seed.
    grid_res        : velocity grid resolution.
    min_vectors_per_cell : discard cells with fewer observations.
    smoothing_sigma : σ for scipy gaussian_filter on U/V.
    crop_padding    : border fraction around the point cloud bounding box.
    keep_sparse_frac : retention rate for sparsest grid cells.
    drop_dense_frac  : retention rate for densest grid cells.
    hexbin_gridsize : hexagonal bin count for panel A.
    top_fitness_pct : percentile threshold for ★ overlay in panel A.
    trail_gens      : how many recent generations to draw as faded trail.
    streamline_density : matplotlib streamplot density parameter.
    n_attractors    : max attractor markers on panel B.
    density_grid_res : KDE grid resolution for panels C/B background.
    density_contour_levels : isolines in panel C.
    fitness_bins    : discrete fitness colour bins.
    cmap_gen        : colormap for generation colouring (default turbo).
    cmap_fit        : colormap for fitness colouring (default fitness_map).
    title_params    : optional dict with keys dataset, gens, pop, noise,
                      algorithm, lambda — shown in the info-box and suptitle.
    figsize         : figure size in inches.
    dpi             : output resolution.
    font_size       : base font size.

    Returns
    -------
    fig : matplotlib Figure.
    """

    # ── 0. rcParams ───────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.size":         font_size,
        "axes.labelsize":    font_size,
        "axes.titlesize":    font_size + 1,
        "xtick.labelsize":   font_size - 1,
        "ytick.labelsize":   font_size - 1,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    if cmap_gen is None:
        cmap_gen = plt.cm.turbo
    if cmap_fit is None:
        cmap_fit = get_discrete_cmap("fitness_map", n=fitness_bins)

    # ── 1. Embeddings & velocity field ────────────────────────────────────
    embedding, gen_labels, per_gen_embeddings = compute_aligned_umap_embedding(
        weights_by_gen, lambda_align=lambda_align, random_state=random_state
    )
    fitness_concat = np.concatenate(fitness_by_gen) if fitness_by_gen else np.array([])

    Xc, Yc, U_raw, V_raw, _ = _compute_velocity_grid(
        per_gen_embeddings,
        grid_res=grid_res,
        min_vectors_per_cell=min_vectors_per_cell,
    )
    U_dense = gaussian_filter(np.ma.filled(U_raw, 0.0).astype(float), sigma=smoothing_sigma)
    V_dense = gaussian_filter(np.ma.filled(V_raw, 0.0).astype(float), sigma=smoothing_sigma)
    speed_dense = np.sqrt(U_dense ** 2 + V_dense ** 2)

    # ── 2. Shared geometry ────────────────────────────────────────────────
    xlim, ylim = auto_crop_to_data(embedding, padding=crop_padding)

    x_1d = Xc[0, :]   # strictly increasing x-centres for streamplot
    y_1d = Yc[:, 0]   # strictly increasing y-centres

    dx_cell = float(Xc[0, 1] - Xc[0, 0]) if Xc.shape[1] > 1 else 1.0
    dy_cell = float(Yc[1, 0] - Yc[0, 0]) if Yc.shape[0] > 1 else 1.0

    # ── 3. Fitness colormap ───────────────────────────────────────────────
    cmap_fit_obj, fit_norm, boundaries = build_fitness_quantile_colormap(
        fitness_concat, n_bins=fitness_bins, cmap_name_or_obj=cmap_fit
    )

    # ── 4. Figure & GridSpec ──────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    gs = GridSpec(
        2, 2, figure=fig,
        hspace=0.15, wspace=0.15,
        left=0.08, right=0.95, top=0.90, bottom=0.07,
    )
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    BG = "#f7f7f7"
    for ax in (ax_A, ax_B, ax_C, ax_D):
        ax.set_facecolor(BG)
        ax.grid(alpha=0.15, linestyle=":", linewidth=0.5, zorder=0)
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
            sp.set_color("#cccccc")

    # ─────────────────────────────────────────────────────────────────────
    # PANEL A — hexbin density + top fitness stars + temporal trail
    # ─────────────────────────────────────────────────────────────────────
    hb = ax_A.hexbin(
        embedding[:, 0], embedding[:, 1],
        C=fitness_concat,
        gridsize=hexbin_gridsize,
        cmap="viridis",
        reduce_C_function=np.mean,   # mean fitness per hex cell
        mincnt=1,
        alpha=0.75,
        linewidths=0.2,
        zorder=2,
    )
    cbar_A = fig.colorbar(hb, ax=ax_A, fraction=0.046, pad=0.04)
    cbar_A.set_label("Mean fitness", labelpad=4)

    # ── top-N% fitness as ★ overlay (never hidden by hexbin) ─────────────
    top_mask = fitness_concat >= np.percentile(fitness_concat, top_fitness_pct)
    if top_mask.sum():
        ax_A.scatter(
            embedding[top_mask, 0], embedding[top_mask, 1],
            c="lime", s=35, marker="*",
            edgecolors="white", linewidths=0.6,
            label=f"Top {100 - top_fitness_pct:.0f}%",
            zorder=10,
        )
        ax_A.legend(
            loc="upper left", fontsize=font_size - 1,
            framealpha=0.7, edgecolor="none",
        )

    # ── temporal trail: last `trail_gens` generations ─────────────────────
    n_gens = len(per_gen_embeddings)
    start  = max(0, n_gens - trail_gens)
    for i, g in enumerate(range(start, n_gens)):
        alpha = 0.15 + 0.65 * (i / max(trail_gens - 1, 1))
        pts   = per_gen_embeddings[g]
        ax_A.scatter(
            pts[:, 0], pts[:, 1],
            c="crimson", s=10, alpha=alpha,
            edgecolors="none", zorder=5,
        )

    ax_A.set_title(
        f"A — Population  (top {100 - top_fitness_pct:.0f}% ★, last {trail_gens} gens trail)",
        fontweight="bold",
    )

    # ─────────────────────────────────────────────────────────────────────
    # PANEL B — streamlines + attractor markers
    # ─────────────────────────────────────────────────────────────────────
    # Faint KDE background for context
    Xg_bg, Yg_bg, kde_bg = _kde_density_grid(embedding, xlim, ylim, grid_res=80)
    ax_B.contourf(
        Xg_bg, Yg_bg, kde_bg,
        levels=6, cmap="Greys", alpha=0.14, zorder=1,
    )

    try:
        strm = ax_B.streamplot(
            x_1d, y_1d,
            U_dense, V_dense,
            density=streamline_density,
            color=speed_dense,
            cmap="viridis",
            linewidth=1.1,
            arrowsize=1.3,
            integration_direction="forward",
            zorder=3,
        )
        cbar_B = fig.colorbar(strm.lines, ax=ax_B, fraction=0.046, pad=0.04)
        cbar_B.set_label("Velocity magnitude", labelpad=4)
    except Exception:
        ax_B.quiver(Xc[::2, ::2], Yc[::2, ::2], U_dense[::2, ::2], V_dense[::2, ::2],
                    color="steelblue", alpha=0.7, zorder=3)

    # ── attractor markers ─────────────────────────────────────────────────
    attractors = _find_attractors(speed_dense, Xc, Yc, n_attractors=n_attractors)
    for ax, ay in attractors:
        ax_B.plot(
            ax, ay,
            marker="*", markersize=18,
            color="crimson", markeredgecolor="white", markeredgewidth=1.8,
            zorder=10,
        )

    ax_B.set_title("B — Streamlines  (▶ attractor★)", fontweight="bold")

    # ─────────────────────────────────────────────────────────────────────
    # PANEL C — time-weighted KDE density (contourf, white background)
    # ─────────────────────────────────────────────────────────────────────
    max_gen_idx = gen_labels.max() if gen_labels.max() > 0 else 1
    weights_t   = 1.0 + 4.0 * (gen_labels / max_gen_idx)
    repeat_cnt  = np.round(weights_t).astype(int).clip(1, 5)
    weighted_pts = np.repeat(embedding, repeat_cnt, axis=0)

    Xg_d, Yg_d, density_w = _kde_density_grid(weighted_pts, xlim, ylim, grid_res=density_grid_res)

    cf_C = ax_C.contourf(
        Xg_d, Yg_d, density_w,
        levels=density_contour_levels,
        cmap="magma",
        norm=mcolors.PowerNorm(gamma=0.5),
        zorder=1,
    )
    ax_C.contour(
        Xg_d, Yg_d, density_w,
        levels=8, colors="white", linewidths=0.5, alpha=0.4, zorder=2,
    )
    cbar_C = fig.colorbar(cf_C, ax=ax_C, fraction=0.046, pad=0.04)
    cbar_C.set_label("Density  (later gens ×5)", labelpad=4)

    # ── convergence basin annotation ──────────────────────────────────────
    max_r, max_c = np.unravel_index(density_w.argmax(), density_w.shape)
    bx, by = float(Xg_d[max_r, max_c]), float(Yg_d[max_r, max_c])
    ax_C.plot(bx, by, "w*", markersize=22, markeredgecolor="black",
              markeredgewidth=1.5, zorder=5)
    ax_C.text(
        bx, by + 0.04 * (ylim[1] - ylim[0]),
        "Basin", ha="center", va="bottom",
        fontsize=font_size, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55),
        zorder=6,
    )

    ax_C.set_title("C — Density  (time-weighted, ★ = basin)", fontweight="bold")

    # ─────────────────────────────────────────────────────────────────────
    # PANEL D — divergence with arrow annotations + solid zero boundary
    # ─────────────────────────────────────────────────────────────────────
    div = compute_divergence(U_dense, V_dense, dx=dx_cell, dy=dy_cell)
    div = gaussian_filter(div, sigma=0.8)   # remove finite-diff edge noise

    div_abs  = float(np.abs(div).max()) or 1.0
    div_norm = mcolors.TwoSlopeNorm(vmin=-div_abs, vcenter=0.0, vmax=div_abs)

    cf_D = ax_D.contourf(
        Xc, Yc, div,
        levels=30, cmap="RdBu_r", norm=div_norm, zorder=1,
    )
    cbar_D = fig.colorbar(cf_D, ax=ax_D, fraction=0.046, pad=0.04)
    cbar_D.set_label("div(F)   [+ explore  /  − exploit]", labelpad=4)

    # solid zero boundary (not dashed — more visible)
    try:
        ax_D.contour(
            Xc, Yc, div,
            levels=[0.0],
            colors="black", linewidths=1.8, linestyles="solid",
            zorder=2,
        )
    except Exception:
        pass

    # ── arrow annotations for peak exploration / exploitation ─────────────
    def _annotate_extreme(ax, grid_x, grid_y, value_grid, is_max: bool):
        """Place an arrow annotation at the extreme cell of value_grid."""
        idx = np.unravel_index(
            value_grid.argmax() if is_max else value_grid.argmin(),
            value_grid.shape,
        )
        px, py = float(grid_x[idx]), float(grid_y[idx])

        # skip if outside the crop window
        if not (xlim[0] < px < xlim[1] and ylim[0] < py < ylim[1]):
            return

        color  = "darkred"   if is_max else "darkblue"
        label  = "Exploration\n(expanding)"  if is_max else "Exploitation\n(converging)"
        offset = (-40, 25)   if is_max else (25, -35)

        ax.annotate(
            label,
            xy=(px, py),
            xytext=offset, textcoords="offset points",
            fontsize=font_size - 1, fontweight="bold", color=color,
            arrowprops=dict(arrowstyle="->", lw=2.0, color=color),
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white", edgecolor=color, linewidth=1.5,
                alpha=0.85,
            ),
            zorder=7,
        )

    _annotate_extreme(ax_D, Xc, Yc, div, is_max=True)
    _annotate_extreme(ax_D, Xc, Yc, div, is_max=False)

    ax_D.set_title("D — Divergence  (explore vs exploit)", fontweight="bold")

    # ─────────────────────────────────────────────────────────────────────
    # Shared axis styling & labels (no duplicate labels)
    # ─────────────────────────────────────────────────────────────────────
    for ax in (ax_A, ax_B, ax_C, ax_D):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # only bottom row gets xlabel
    for ax in (ax_A, ax_B):
        ax.tick_params(labelbottom=False)
    for ax in (ax_C, ax_D):
        ax.set_xlabel("UMAP-1", fontweight="bold")

    # only left column gets ylabel
    for ax in (ax_A, ax_C):
        ax.set_ylabel("UMAP-2", fontweight="bold")
    for ax in (ax_B, ax_D):
        ax.tick_params(labelleft=False)

    # ─────────────────────────────────────────────────────────────────────
    # Metadata info-box (monospace, top-right)
    # ─────────────────────────────────────────────────────────────────────
    if title_params is None:
        n_gen_val = len(weights_by_gen)
        pop_val   = len(weights_by_gen[0]) if weights_by_gen else 0
        title_params = {
            "dataset":   "—",
            "gens":       n_gen_val,
            "pop":        pop_val,
            "noise":      float("nan"),
            "algorithm": "Simple Gaussian ES",
            "lambda":     lambda_align,
        }

    best_fit = float(fitness_concat.max()) if len(fitness_concat) else float("nan")
    info = (
        f"Dataset   : {title_params.get('dataset', '—')}\n"
        f"Algorithm : {title_params.get('algorithm', '—')}\n"
        f"Gens      : {title_params.get('gens', '—')}   "
        f"Pop : {title_params.get('pop', '—')}\n"
        f"σ_mut     : {title_params.get('noise', float('nan')):.3f}   "
        f"λ_align : {title_params.get('lambda', lambda_align):.2f}\n"
        f"Best fit  : {best_fit:.4f}"
    )
    fig.text(
        0.965, 0.955, info,
        transform=fig.transFigure,
        fontsize=font_size - 1, family="monospace",
        va="top", ha="right",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="wheat", edgecolor="#888", linewidth=1.0, alpha=0.85,
        ),
    )

    # ── suptitle ──────────────────────────────────────────────────────────
    suptitle = (
        f"Weight-Space Dynamics — {title_params.get('dataset', '')}  "
        f"({title_params.get('gens', '')} gens, pop {title_params.get('pop', '')},  "
        f"σ={title_params.get('noise', float('nan')):.2f})\n"
        f"{title_params.get('algorithm', '')}   λ_align={title_params.get('lambda', lambda_align):.2f}"
    )
    fig.suptitle(suptitle, fontsize=font_size + 3, fontweight="bold", y=0.99)

    return fig
