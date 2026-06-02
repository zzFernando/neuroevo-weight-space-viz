"""
UMAP + velocity-field analysis for a saved neuroevolution run.
Implements the same pipeline as the paper:
  1. Joint UMAP embedding of all individuals from all generations.
  2. Reference-anchored interpolation toward generation 0 with strength lambda.
  3. Per-generation grid-binned velocity field.

Usage:
  python analyze_umap.py runs/halfcheetah_seed42.npz \\
      --lambda_align 0.8 --out figures/halfcheetah_seed42.png
"""

from __future__ import annotations
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
import umap


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def joint_umap(populations, n_neighbors=15, min_dist=0.1, seed=0):
    """populations: (gens, pop, n_params)  ->  Z: (gens, pop, 2)."""
    G, P, D = populations.shape
    flat = populations.reshape(G * P, D)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=seed)
    Z = reducer.fit_transform(flat)
    return Z.reshape(G, P, 2)


def align_to_reference(Z, lam=0.8, ref_idx=0):
    """Eq. 1 of the paper:  Z_tilde_k = (1 - lam) Z_k + lam Z_ref."""
    Z_ref = Z[ref_idx]
    return (1 - lam) * Z + lam * Z_ref[None, :, :]


# ---------------------------------------------------------------------------
# Velocity field on a grid (Section 3.3 of the paper)
# ---------------------------------------------------------------------------

def velocity_field(Z_aligned, n_bins=22):
    G, P, _ = Z_aligned.shape
    pts = Z_aligned.reshape(-1, 2)
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    # Add a small margin
    pad = 0.05 * (x_max - x_min)
    x_edges = np.linspace(x_min - pad, x_max + pad, n_bins + 1)
    y_edges = np.linspace(y_min - pad, y_max + pad, n_bins + 1)

    U = np.zeros((n_bins, n_bins))
    V = np.zeros((n_bins, n_bins))
    counts = np.zeros((n_bins, n_bins))

    for g in range(G - 1):
        # displacement of each individual from gen g to gen g+1
        z0 = Z_aligned[g]
        z1 = Z_aligned[g + 1]
        # midpoint binning (centroid falls in cell)
        mid = 0.5 * (z0 + z1)
        ix = np.clip(np.searchsorted(x_edges, mid[:, 0]) - 1, 0, n_bins - 1)
        iy = np.clip(np.searchsorted(y_edges, mid[:, 1]) - 1, 0, n_bins - 1)
        delta = z1 - z0
        for k in range(z0.shape[0]):
            U[iy[k], ix[k]] += delta[k, 0]
            V[iy[k], ix[k]] += delta[k, 1]
            counts[iy[k], ix[k]] += 1

    mask = counts > 0
    U[mask] /= counts[mask]
    V[mask] /= counts[mask]

    # Light 3x3 box smoothing (paper does this before plotting)
    def smooth(M):
        from scipy.ndimage import uniform_filter
        return uniform_filter(M, size=3)
    return x_edges, y_edges, smooth(U), smooth(V), counts


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_panel(Z, fits, x_edges, y_edges, U, V, out_path, title=''):
    G, P, _ = Z.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    pts = Z.reshape(-1, 2)
    gens = np.repeat(np.arange(G), P)
    fits_flat = fits.reshape(-1)

    # 1) generation coloring
    sc0 = axes[0].scatter(pts[:, 0], pts[:, 1], c=gens, cmap='viridis',
                          s=4, alpha=0.6)
    axes[0].set_title('Aligned UMAP — by generation')
    plt.colorbar(sc0, ax=axes[0], label='generation')

    # 2) fitness coloring
    sc1 = axes[1].scatter(pts[:, 0], pts[:, 1], c=fits_flat, cmap='plasma',
                          s=4, alpha=0.6)
    axes[1].set_title('Aligned UMAP — by fitness')
    plt.colorbar(sc1, ax=axes[1], label='fitness')

    # 3) velocity field
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xc, Yc = np.meshgrid(xc, yc)
    speed = np.sqrt(U ** 2 + V ** 2)
    axes[2].streamplot(Xc, Yc, U, V, color=speed, cmap='magma', density=1.4,
                       linewidth=1.0)
    axes[2].scatter(pts[:, 0], pts[:, 1], c='lightgray', s=1, alpha=0.2,
                    zorder=0)
    axes[2].set_title('Population velocity field')

    for ax in axes:
        ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
        ax.set_aspect('equal', adjustable='datalim')

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] saved -> {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('npz')
    p.add_argument('--lambda_align', type=float, default=0.8)
    p.add_argument('--n_neighbors',  type=int,   default=15)
    p.add_argument('--min_dist',     type=float, default=0.1)
    p.add_argument('--umap_seed',    type=int,   default=0)
    p.add_argument('--out',          default=None)
    args = p.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    pops = data['populations']      # (gens, pop, n_params)
    fits = data['fitnesses']        # (gens, pop)
    meta = data['meta'].item() if 'meta' in data.files else {}
    print(f'[load] populations shape={pops.shape}, fitness shape={fits.shape}')
    print(f'[load] meta={meta}')

    Z = joint_umap(pops, args.n_neighbors, args.min_dist, args.umap_seed)
    Z_aligned = align_to_reference(Z, lam=args.lambda_align)
    x_e, y_e, U, V, _ = velocity_field(Z_aligned, n_bins=22)

    out = args.out or args.npz.replace('.npz', '.png')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    plot_panel(Z_aligned, fits, x_e, y_e, U, V, out,
               title=f'{meta.get("env", "env")} — seed {meta.get("seed", "?")} '
                     f'— $\\lambda$={args.lambda_align}')


if __name__ == '__main__':
    main()
