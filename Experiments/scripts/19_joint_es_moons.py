"""Exp 19: the Exp 17/18 joint-ES UMAP study, replicated on make_moons.

Uses the evosax runs produced by brax/moons_evosax.py (48-dim weight space,
runs/moons_{algo}_seed{seed}.npz). Same joint-fit methodology: pool all algorithms
into one UMAP so their search geometries are directly comparable.

Outputs:
  exp19_moons_algo_vs_fitness.png  — joint UMAP: by algorithm | by fitness
  exp19_moons_multiseed.png        — joint UMAP per seed (42, 7, 123)
  results/exp19_moons_signatures.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap


from shared import (
    CACHE_DIR, FIGURES_DIR, RESULTS_DIR, ensure_dirs, set_science_style,
    count_attractors_dbscan, save_results_csv,
)

ALGOS = ["simple_ga", "open_es", "cma_es", "sep_cma_es"]
ALGO_LABELS = {"simple_ga": "Simple GA", "open_es": "OpenES",
               "cma_es": "CMA-ES", "sep_cma_es": "sep-CMA-ES"}
ALGO_COLORS = {"simple_ga": "#1f77b4", "open_es": "#ff7f0e",
               "cma_es": "#2ca02c", "sep_cma_es": "#d62728"}
SEEDS = [42, 7, 123]
from paths import RUNS_DIR as BRAX_RUNS


def load_runs(seed: int):
    W, names, fit = [], [], []
    for algo in ALGOS:
        f = BRAX_RUNS / f"moons_{algo}_seed{seed}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        pops = d["populations"].astype(np.float64)
        fits = d["fitnesses"].astype(float)
        G, P, _ = pops.shape
        W.append(pops.reshape(G * P, -1))
        names += [algo] * (G * P)
        fit.append(fits.reshape(-1))
    return np.vstack(W), np.array(names), np.concatenate(fit)


def fit_umap(W, key, seed=42, n_components=2):
    p = CACHE_DIR / f"exp19_{key}_{n_components}d.npz"
    if p.exists():
        return np.load(p)["emb"]
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=n_components,
                    metric="euclidean", random_state=seed).fit_transform(W)
    np.savez_compressed(p, emb=emb)
    return emb


def scatter_algo(ax, emb, names):
    ax.set_facecolor("white")
    for algo in ALGOS:
        m = names == algo
        ax.scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.45, edgecolors="none",
                   color=ALGO_COLORS[algo], label=ALGO_LABELS[algo], rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])


def legend_algo(ax):
    ax.legend([plt.Line2D([0], [0], marker="o", ls="", color=ALGO_COLORS[a]) for a in ALGOS],
              [ALGO_LABELS[a] for a in ALGOS], markerscale=1.4, fontsize=8, loc="best")


def fig_algo_vs_fitness(seed=42):
    W, names, fit = load_runs(seed)
    emb = fit_umap(W, f"joint_s{seed}", seed)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    scatter_algo(axes[0], emb, names)
    legend_algo(axes[0])
    axes[0].set_title("colored by algorithm", fontsize=10)

    fmin, fmax = np.percentile(fit, 50), np.percentile(fit, 99)
    c = np.clip((fit - fmin) / (fmax - fmin + 1e-9) * 100, 0, 100)
    sc = axes[1].scatter(emb[:, 0], emb[:, 1], c=c, cmap="viridis", s=3, alpha=0.55,
                         edgecolors="none", rasterized=True)
    best = int(np.argmax(fit))
    axes[1].scatter(emb[best, 0], emb[best, 1], marker="*", s=160, c="#d62728",
                    edgecolors="white", linewidths=0.8, zorder=5)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].set_title("colored by fitness  (★ = global best)", fontsize=10)
    fig.colorbar(sc, ax=axes[1], label="fitness percentile", shrink=0.75, pad=0.02)

    fig.suptitle(f"Joint UMAP of 4 ES algorithms on make_moons "
                 f"(48-dim weight space, seed {seed})", fontsize=12, y=1.0)
    out = FIGURES_DIR / "exp19_moons_algo_vs_fitness.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def fig_multiseed():
    fig, axes = plt.subplots(1, len(SEEDS), figsize=(5.2 * len(SEEDS), 5.2))
    for ax, seed in zip(np.atleast_1d(axes), SEEDS):
        W, names, _ = load_runs(seed)
        emb = fit_umap(W, f"joint_s{seed}", seed)
        scatter_algo(ax, emb, names)
        ax.set_title(f"seed {seed}", fontsize=10)
    legend_algo(np.atleast_1d(axes)[0])
    fig.suptitle("make_moons — joint UMAP per seed (48-dim, euclidean)", fontsize=12, y=1.0)
    out = FIGURES_DIR / "exp19_moons_multiseed.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def quantify(seed=42):
    W, names, fit = load_runs(seed)
    emb = fit_umap(W, f"joint_s{seed}", seed)
    rows = []
    for algo in ALGOS:
        m = names == algo
        e = emb[m]
        compact = float(np.linalg.norm(e - e.mean(0), axis=1).mean())
        bbox = float((e[:, 0].max() - e[:, 0].min()) * (e[:, 1].max() - e[:, 1].min()))
        rows.append({
            "algorithm": ALGO_LABELS[algo],
            "n_clusters": count_attractors_dbscan(e, min_samples=5),
            "compactness_mean_dist": round(compact, 3),
            "coverage_bbox_area": round(bbox, 1),
            "best_fitness": round(float(fit[m].max()), 3),
        })
    save_results_csv(rows, RESULTS_DIR / "exp19_moons_signatures.csv")
    print("\nmake_moons search-geometry signatures (seed 42):")
    print(f"  {'algorithm':12s} {'clusters':>9s} {'compactness':>12s} {'coverage':>10s} {'best_fit':>9s}")
    for r in rows:
        print(f"  {r['algorithm']:12s} {r['n_clusters']:9d} {r['compactness_mean_dist']:12.3f} "
              f"{r['coverage_bbox_area']:10.1f} {r['best_fitness']:9.3f}")


def fig_3d(seed=42):
    W, names, fit = load_runs(seed)
    emb = fit_umap(W, f"joint_s{seed}", seed, n_components=3)
    fig = plt.figure(figsize=(14, 6.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.set_facecolor("white")
    for algo in ALGOS:
        m = names == algo
        ax1.scatter(emb[m, 0], emb[m, 1], emb[m, 2], s=3, alpha=0.45, edgecolors="none",
                    color=ALGO_COLORS[algo], label=ALGO_LABELS[algo])
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])
    ax1.legend(markerscale=1.4, fontsize=8, loc="best")
    ax1.set_title("by algorithm", fontsize=10)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    fmin, fmax = np.percentile(fit, 50), np.percentile(fit, 99)
    c = np.clip((fit - fmin) / (fmax - fmin + 1e-9) * 100, 0, 100)
    sc = ax2.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=c, cmap="viridis", s=4, alpha=0.5, edgecolors="none")
    best = int(np.argmax(fit))
    ax2.scatter(emb[best, 0], emb[best, 1], emb[best, 2], marker="*", s=180,
                c="#d62728", edgecolors="white", linewidths=0.8)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])
    ax2.set_title("by fitness (★ = best)", fontsize=10)
    fig.colorbar(sc, ax=ax2, label="fitness percentile", shrink=0.5, pad=0.08)
    fig.suptitle(f"3-D joint UMAP of 4 ES on make_moons (seed {seed})", fontsize=12, y=1.02)
    out = FIGURES_DIR / "exp19_moons_joint_3d.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def gif_3d(seed=42, n_frames=60):
    from matplotlib.animation import FuncAnimation, PillowWriter
    W, names, _ = load_runs(seed)
    emb = fit_umap(W, f"joint_s{seed}", seed, n_components=3)
    fig = plt.figure(figsize=(7, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    for algo in ALGOS:
        m = names == algo
        ax.scatter(emb[m, 0], emb[m, 1], emb[m, 2], s=3, alpha=0.45, edgecolors="none",
                   color=ALGO_COLORS[algo], label=ALGO_LABELS[algo])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.legend(markerscale=1.4, fontsize=8, loc="best")
    ax.set_title(f"3-D joint UMAP of 4 ES on make_moons (seed {seed})", fontsize=10)

    def update(frame):
        ax.view_init(elev=20, azim=frame * (360 / n_frames))
        return ()

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    out = FIGURES_DIR / "exp19_moons_joint_3d.gif"
    anim.save(out, writer=PillowWriter(fps=15), dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    ensure_dirs()
    set_science_style()
    fig_algo_vs_fitness()
    fig_multiseed()
    fig_3d()
    gif_3d()
    quantify()


if __name__ == "__main__":
    main()
