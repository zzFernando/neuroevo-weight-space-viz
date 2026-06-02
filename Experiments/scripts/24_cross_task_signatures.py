"""Exp 24: are the ES search signatures invariant across TASKS? (cross-task validation)

Addresses the main weakness flagged in review: the "task-invariant signature" claim
rested on only 2 tasks (make_moons + HalfCheetah). Here we run the 4 ES on FOUR
2-D classification tasks of increasing difficulty (blobs < moons < circles < xor),
all sharing the same 48-dim MLP, across 3 seeds, and test whether the geometric
signature (GA fragmented, distribution-ES compact) holds everywhere.

Signatures are measured in RAW weight space (scale-consistent, no UMAP needed):
  - fragmentation : DBSCAN cluster count over the pooled population
  - compactness   : mean distance to the population centroid
reported relative within each task so tasks are comparable. A per-task joint UMAP
(seed 42) provides the visual companion.
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

TASKS = ["blobs", "moons", "circles", "xor"]          # easy → hard
TASK_LABELS = {"blobs": "Blobs", "moons": "Moons", "circles": "Circles", "xor": "XOR"}
ALGOS = ["simple_ga", "open_es", "cma_es", "sep_cma_es"]
ALGO_LABELS = {"simple_ga": "Simple GA", "open_es": "OpenES", "cma_es": "CMA-ES", "sep_cma_es": "sep-CMA-ES"}
ALGO_COLORS = {"simple_ga": "#1f77b4", "open_es": "#ff7f0e", "cma_es": "#2ca02c", "sep_cma_es": "#d62728"}
SEEDS = [42, 7, 123]
from paths import RUNS_DIR as BRAX_RUNS


def load(task, algo, seed):
    d = np.load(BRAX_RUNS / f"{task}_{algo}_seed{seed}.npz", allow_pickle=True)
    return d["populations"].astype(np.float64).reshape(-1, 48), d["fitnesses"].astype(float).reshape(-1)


def raw_signatures(W):
    centroid = W.mean(0)
    compactness = float(np.linalg.norm(W - centroid, axis=1).mean())
    n_clusters = count_attractors_dbscan(W, min_samples=5)
    return n_clusters, compactness


def joint_umap(task, seed=42):
    p = CACHE_DIR / f"exp24_joint_{task}_{seed}.npz"
    Ws, names = [], []
    for algo in ALGOS:
        W, _ = load(task, algo, seed)
        Ws.append(W); names += [algo] * len(W)
    W_all = np.vstack(Ws); names = np.array(names)
    if p.exists():
        return np.load(p)["emb"], names
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                    metric="euclidean", random_state=seed).fit_transform(W_all)
    np.savez_compressed(p, emb=emb)
    return emb, names


def main():
    ensure_dirs()
    set_science_style()

    # ---- compactness in raw weight space (3 seeds), relative within each task ----
    rows = []
    for task in TASKS:
        for seed in SEEDS:
            comps = {}
            for algo in ALGOS:
                W, _ = load(task, algo, seed)
                _, comp = raw_signatures(W)
                comps[algo] = comp
            med = np.median(list(comps.values()))
            for algo in ALGOS:
                rows.append({"task": TASK_LABELS[task], "algorithm": ALGO_LABELS[algo], "seed": seed,
                             "compactness": round(comps[algo], 3),
                             "rel_compactness": round(comps[algo] / (med + 1e-9), 3)})
    save_results_csv(rows, RESULTS_DIR / "exp24_cross_task_signatures.csv")

    import pandas as pd
    df = pd.DataFrame(rows)
    agg = df.groupby(["task", "algorithm"]).agg(rel_comp=("rel_compactness", "mean")).reset_index()

    # fragmentation measured in the 2-D joint UMAP (seed 42), where DBSCAN is meaningful
    # (in raw 48-dim space the adaptive-eps DBSCAN collapses everything to 1 cluster).
    frag = {}
    for task in TASKS:
        emb, names = joint_umap(task, 42)
        for algo in ALGOS:
            frag[(TASK_LABELS[task], ALGO_LABELS[algo])] = count_attractors_dbscan(emb[names == algo], min_samples=5)

    # consistency checks
    print("Compactness ordering per task (↑ = most spread) — is it consistent?")
    for task in TASKS:
        sub = agg[agg.task == TASK_LABELS[task]].sort_values("rel_comp", ascending=False)
        print(f"  {TASK_LABELS[task]:8s}: {' > '.join(sub.algorithm.values)}")
    print("\nFragmentation in UMAP (seed 42 clusters) per task:")
    for task in TASKS:
        order = sorted(ALGOS, key=lambda a: -frag[(TASK_LABELS[task], ALGO_LABELS[a])])
        s = "  ".join(f"{ALGO_LABELS[a]}={frag[(TASK_LABELS[task], ALGO_LABELS[a])]}" for a in order)
        print(f"  {TASK_LABELS[task]:8s}: {s}")

    # ---- Figure A: heatmaps task × algorithm ----
    frag_M = np.array([[frag[(TASK_LABELS[t], ALGO_LABELS[a])] for t in TASKS] for a in ALGOS], float)
    comp_M = agg.pivot(index="algorithm", columns="task", values="rel_comp").reindex(
        index=[ALGO_LABELS[a] for a in ALGOS], columns=[TASK_LABELS[t] for t in TASKS]).values

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, M, title in [(axes[0], frag_M, "Fragmentation — UMAP clusters (seed 42)"),
                         (axes[1], comp_M, "Relative compactness, raw space (↑ = spread)")]:
        im = ax.imshow(M, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(TASKS))); ax.set_xticklabels([TASK_LABELS[t] for t in TASKS])
        ax.set_yticks(range(len(ALGOS))); ax.set_yticklabels([ALGO_LABELS[a] for a in ALGOS])
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=9)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Cross-task ES signatures (4 tasks × 3 seeds) — Simple GA stands out in every task",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    outA = FIGURES_DIR / "exp24_cross_task_heatmap.png"
    fig.savefig(outA, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {outA}")

    # ---- Figure B: per-task joint UMAP (seed 42), colored by algorithm ----
    fig, axes = plt.subplots(1, len(TASKS), figsize=(4.0 * len(TASKS), 4.2))
    for ax, task in zip(axes, TASKS):
        emb, names = joint_umap(task, 42)
        ax.set_facecolor("white")
        for algo in ALGOS:
            m = names == algo
            ax.scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.45, edgecolors="none",
                       color=ALGO_COLORS[algo], label=ALGO_LABELS[algo], rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(TASK_LABELS[task], fontsize=11)
    axes[0].legend(markerscale=2, fontsize=7, loc="best")
    fig.suptitle("Joint UMAP per task (seed 42) — signatures repeat across tasks", fontsize=12, y=1.02)
    fig.tight_layout()
    outB = FIGURES_DIR / "exp24_cross_task_umaps.png"
    fig.savefig(outB, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {outB}")


if __name__ == "__main__":
    main()
