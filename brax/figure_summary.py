"""
Composite figure for the HalfCheetah benchmark:
  - Fitness curves (mean / best, per seed)
  - Velocity fields per seed, side-by-side
Reads multiple .npz files from runs/ and produces a single PNG.
"""
import argparse, os, glob
import numpy as np
import matplotlib.pyplot as plt
import umap


def joint_umap(populations, n_neighbors=15, min_dist=0.1, seed=0):
    G, P, D = populations.shape
    flat = populations.reshape(G * P, D)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=seed)
    return reducer.fit_transform(flat).reshape(G, P, 2)


def align(Z, lam=0.8):
    return (1 - lam) * Z + lam * Z[0][None]


def vf(Z, n_bins=22):
    from scipy.ndimage import uniform_filter
    G, P, _ = Z.shape
    pts = Z.reshape(-1, 2)
    pad = 0.05 * (pts[:, 0].max() - pts[:, 0].min())
    x_edges = np.linspace(pts[:, 0].min() - pad, pts[:, 0].max() + pad, n_bins + 1)
    y_edges = np.linspace(pts[:, 1].min() - pad, pts[:, 1].max() + pad, n_bins + 1)
    U = np.zeros((n_bins, n_bins)); V = np.zeros((n_bins, n_bins))
    C = np.zeros((n_bins, n_bins))
    for g in range(G - 1):
        z0, z1 = Z[g], Z[g + 1]
        mid = 0.5 * (z0 + z1)
        ix = np.clip(np.searchsorted(x_edges, mid[:, 0]) - 1, 0, n_bins - 1)
        iy = np.clip(np.searchsorted(y_edges, mid[:, 1]) - 1, 0, n_bins - 1)
        d = z1 - z0
        for k in range(z0.shape[0]):
            U[iy[k], ix[k]] += d[k, 0]; V[iy[k], ix[k]] += d[k, 1]; C[iy[k], ix[k]] += 1
    m = C > 0
    U[m] /= C[m]; V[m] /= C[m]
    return x_edges, y_edges, uniform_filter(U, 3), uniform_filter(V, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs_glob', default='runs/halfcheetah_seed*.npz')
    p.add_argument('--out',       default='figures/halfcheetah_summary.png')
    args = p.parse_args()

    paths = sorted(glob.glob(args.runs_glob))
    print(f'Found {len(paths)} runs: {paths}')

    runs = []
    for path in paths:
        d = np.load(path, allow_pickle=True)
        runs.append({'path': path,
                     'pops': d['populations'],
                     'fits': d['fitnesses'],
                     'meta': d['meta'].item()})

    n = len(runs)
    fig = plt.figure(figsize=(5 * n, 9))
    gs = fig.add_gridspec(2, n, height_ratios=[1, 1.4])

    # Row 0: fitness curves (one panel spanning all columns)
    ax_fit = fig.add_subplot(gs[0, :])
    for r in runs:
        gens = np.arange(r['fits'].shape[0])
        ax_fit.plot(gens, r['fits'].mean(axis=1),
                    label=f"seed {r['meta']['seed']} (mean)")
        ax_fit.plot(gens, r['fits'].max(axis=1), '--', alpha=0.5,
                    label=f"seed {r['meta']['seed']} (best)")
    ax_fit.set_xlabel('generation'); ax_fit.set_ylabel('fitness (episode return)')
    ax_fit.set_title('HalfCheetah — fitness over generations')
    ax_fit.legend(ncol=2, fontsize=9); ax_fit.grid(alpha=0.3)

    # Row 1: velocity field per seed
    for i, r in enumerate(runs):
        Z = align(joint_umap(r['pops'], seed=0))
        xe, ye, U, V = vf(Z)
        xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
        Xc, Yc = np.meshgrid(xc, yc)
        speed = np.sqrt(U ** 2 + V ** 2)

        ax = fig.add_subplot(gs[1, i])
        pts = Z.reshape(-1, 2)
        fits = r['fits'].reshape(-1)
        ax.scatter(pts[:, 0], pts[:, 1], c=fits, cmap='plasma', s=4, alpha=0.45)
        ax.streamplot(Xc, Yc, U, V, color=speed, cmap='magma',
                      density=1.3, linewidth=0.9)
        ax.set_title(f"seed {r['meta']['seed']} — velocity field "
                     f"+ fitness coloring")
        ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
        ax.set_aspect('equal', adjustable='datalim')

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches='tight')
    print(f'saved -> {args.out}')


if __name__ == '__main__':
    main()
