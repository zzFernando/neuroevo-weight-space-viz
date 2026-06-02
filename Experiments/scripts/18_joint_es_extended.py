"""Exp 18: extended joint-UMAP study of neuroevolution algorithms.

Builds on Exp 17 with three deliverables:
  (a) multi-seed consistency  — joint UMAP per seed (42, 7, 123), by algorithm,
      to check the geometric search signatures are stable, not seed artifacts.
  (b) long-run trajectory     — add open_es_p128g500 (500 gen × 128 pop) to the
      seed-42 joint fit to see OpenES's full path.
  (c) 3-D joint embedding     — seed-42 joint UMAP in 3-D, by algorithm + fitness.

All fits are euclidean on the shared 390-dim HalfCheetah weight space, fixed seed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap


from shared import CACHE_DIR, FIGURES_DIR, ensure_dirs, set_science_style

ALGOS = ["simple_ga", "open_es", "cma_es", "sep_cma_es"]
ALGO_LABELS = {"simple_ga": "Simple GA", "open_es": "OpenES", "cma_es": "CMA-ES",
               "sep_cma_es": "sep-CMA-ES", "open_es_p128g500": "OpenES (long)"}
ALGO_COLORS = {"simple_ga": "#1f77b4", "open_es": "#ff7f0e", "cma_es": "#2ca02c",
               "sep_cma_es": "#d62728", "open_es_p128g500": "#9467bd"}
SEEDS = [42, 7, 123]
from paths import RUNS_DIR as BRAX_RUNS


def load_runs(specs: list[tuple[str, int]], max_per_run: int | None = None):
    """specs = [(algo, seed), ...] -> (W, algo_names, gen_idx, fit).

    max_per_run subsamples each run to that many individuals (fixed RNG), so runs
    of very different sizes contribute comparable density to a joint fit.
    """
    rng = np.random.default_rng(0)
    W, names, gen_idx, fit = [], [], [], []
    for algo, seed in specs:
        f = BRAX_RUNS / f"{algo}_seed{seed}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        pops = d["populations"].astype(np.float64)
        fits = d["fitnesses"].astype(float)
        G, P, _ = pops.shape
        Wr = pops.reshape(G * P, -1)
        gr = np.repeat(np.arange(G), P)
        fr = fits.reshape(-1)
        if max_per_run is not None and len(Wr) > max_per_run:
            sel = rng.choice(len(Wr), max_per_run, replace=False)
            Wr, gr, fr = Wr[sel], gr[sel], fr[sel]
        W.append(Wr)
        names += [algo] * len(Wr)
        gen_idx.append(gr)
        fit.append(fr)
    return (np.vstack(W), np.array(names), np.concatenate(gen_idx), np.concatenate(fit))


def fit_umap(W: np.ndarray, key: str, n_components: int = 2, seed: int = 42) -> np.ndarray:
    p = CACHE_DIR / f"exp18_{key}_{n_components}d.npz"
    if p.exists():
        return np.load(p)["emb"]
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=n_components,
                    metric="euclidean", random_state=seed).fit_transform(W)
    np.savez_compressed(p, emb=emb)
    return emb


def scatter_algo(ax, emb, names, algos, dims=(0, 1)):
    ax.set_facecolor("white")
    for algo in algos:
        m = names == algo
        ax.scatter(*[emb[m, d] for d in dims], s=3, alpha=0.45, edgecolors="none",
                   color=ALGO_COLORS[algo], label=ALGO_LABELS[algo], rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    if hasattr(ax, "set_zticks"):
        ax.set_zticks([])


def legend_algo(ax, algos):
    leg = ax.legend([plt.Line2D([0], [0], marker="o", ls="", color=ALGO_COLORS[a],
                                label=ALGO_LABELS[a]) for a in algos],
                    [ALGO_LABELS[a] for a in algos], markerscale=1.4, fontsize=8, loc="best")
    return leg


# ---------------------------------------------------------------------------
# (a) multi-seed consistency
# ---------------------------------------------------------------------------
def fig_multiseed():
    fig, axes = plt.subplots(1, len(SEEDS), figsize=(5.2 * len(SEEDS), 5.2))
    for ax, seed in zip(np.atleast_1d(axes), SEEDS):
        W, names, _, _ = load_runs([(a, seed) for a in ALGOS])
        emb = fit_umap(W, f"multiseed_s{seed}", 2, seed)
        scatter_algo(ax, emb, names, ALGOS)
        ax.set_title(f"seed {seed}", fontsize=10)
    legend_algo(np.atleast_1d(axes)[0], ALGOS)
    fig.suptitle("Joint UMAP per seed — are the search signatures consistent?  "
                 "(HalfCheetah, 390-dim, euclidean)", fontsize=12, y=1.0)
    out = FIGURES_DIR / "exp18a_multiseed_consistency.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# (b) long-run trajectory
# ---------------------------------------------------------------------------
def fig_longrun(seed: int = 42, max_per_run: int = 4000):
    """Subsample every run to max_per_run so the 64k-point long run no longer
    dominates the fit and hides the others."""
    algos = ALGOS + ["open_es_p128g500"]
    W, names, gen_idx, _ = load_runs([(a, seed) for a in algos], max_per_run=max_per_run)
    emb = fit_umap(W, f"longrun_s{seed}_sub{max_per_run}", 2, seed)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    scatter_algo(ax, emb, names, algos)
    legend_algo(ax, algos)
    ax.set_title(f"Joint UMAP incl. OpenES long run — balanced ({max_per_run}/run)\n"
                 f"HalfCheetah, seed {seed}, euclidean", fontsize=10)
    out = FIGURES_DIR / "exp18b_longrun_trajectory.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}  (pooled {W.shape[0]} individuals, balanced)")


# ---------------------------------------------------------------------------
# (c) 3-D joint embedding
# ---------------------------------------------------------------------------
def fig_3d(seed: int = 42):
    W, names, _, fit = load_runs([(a, seed) for a in ALGOS])
    emb = fit_umap(W, f"3d_s{seed}", 3, seed)

    fig = plt.figure(figsize=(14, 6.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    scatter_algo(ax1, emb, names, ALGOS, dims=(0, 1, 2))
    legend_algo(ax1, ALGOS)
    ax1.set_title("by algorithm", fontsize=10)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    fmin, fmax = np.percentile(fit, 50), np.percentile(fit, 99)
    c = np.clip((fit - fmin) / (fmax - fmin + 1e-9) * 100, 0, 100)
    sc = ax2.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=c, cmap="viridis",
                     s=4, alpha=0.5, edgecolors="none")
    best = int(np.argmax(fit))
    ax2.scatter(emb[best, 0], emb[best, 1], emb[best, 2], marker="*", s=180,
                c="#d62728", edgecolors="white", linewidths=0.8)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])
    ax2.set_title("by fitness (★ = best)", fontsize=10)
    fig.colorbar(sc, ax=ax2, label="fitness percentile", shrink=0.5, pad=0.08)

    fig.suptitle(f"3-D joint UMAP of 4 ES algorithms (HalfCheetah, seed {seed})",
                 fontsize=12, y=1.02)
    out = FIGURES_DIR / "exp18c_joint_3d.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def gif_3d(seed: int = 42, n_frames: int = 60):
    """Rotating GIF of the 3-D joint embedding, colored by algorithm."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    W, names, _, _ = load_runs([(a, seed) for a in ALGOS])
    emb = fit_umap(W, f"3d_s{seed}", 3, seed)

    fig = plt.figure(figsize=(7, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    scatter_algo(ax, emb, names, ALGOS, dims=(0, 1, 2))
    legend_algo(ax, ALGOS)
    ax.set_title(f"3-D joint UMAP of 4 ES algorithms (seed {seed})", fontsize=10)

    def update(frame):
        ax.view_init(elev=20, azim=frame * (360 / n_frames))
        return ()

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    out = FIGURES_DIR / "exp18c_joint_3d.gif"
    anim.save(out, writer=PillowWriter(fps=15), dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


def quantify_signatures(seed: int = 42):
    """Quantitative fingerprint of each algorithm's search geometry (2-D joint emb)."""
    from shared import count_attractors_dbscan, save_results_csv, RESULTS_DIR

    W, names, _, fit = load_runs([(a, seed) for a in ALGOS])
    emb = fit_umap(W, f"multiseed_s{seed}", 2, seed)  # reuse seed-42 2-D fit

    rows = []
    for algo in ALGOS:
        m = names == algo
        e = emb[m]
        centroid = e.mean(0)
        compactness = float(np.linalg.norm(e - centroid, axis=1).mean())  # mean dist to centroid
        bbox = float((e[:, 0].max() - e[:, 0].min()) * (e[:, 1].max() - e[:, 1].min()))
        n_clusters = count_attractors_dbscan(e, min_samples=5)
        rows.append({
            "algorithm": ALGO_LABELS[algo],
            "n_clusters": n_clusters,
            "compactness_mean_dist": round(compactness, 3),
            "coverage_bbox_area": round(bbox, 1),
            "best_fitness": round(float(fit[m].max()), 1),
        })

    save_results_csv(rows, RESULTS_DIR / "exp18_signatures.csv")
    print("\nSearch-geometry signatures (seed 42, 2-D joint embedding):")
    print(f"  {'algorithm':12s} {'clusters':>9s} {'compactness':>12s} {'coverage':>10s} {'best_fit':>9s}")
    for r in rows:
        print(f"  {r['algorithm']:12s} {r['n_clusters']:9d} {r['compactness_mean_dist']:12.3f} "
              f"{r['coverage_bbox_area']:10.1f} {r['best_fitness']:9.1f}")


def main():
    ensure_dirs()
    set_science_style()
    fig_multiseed()
    fig_longrun()
    fig_3d()
    gif_3d()
    quantify_signatures()


if __name__ == "__main__":
    main()
