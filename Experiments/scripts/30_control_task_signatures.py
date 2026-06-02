"""Exp 30: do the ES search signatures hold on CONTROL tasks?

Extends the cross-task validation (Exp 24, classification) to control / RL tasks:
CartPole (d_w=80, classic dynamics) and HalfCheetah (d_w=390, brax). For each task we
fit a joint UMAP of the 4 ES and measure the signature (UMAP fragmentation + raw-space
compactness). If Simple GA fragments and distribution-ES stay compact here too, the
signature is invariant across the classification↔control divide — 6 tasks total.
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
ALGO_LABELS = {"simple_ga": "Simple GA", "open_es": "OpenES", "cma_es": "CMA-ES", "sep_cma_es": "sep-CMA-ES"}
ALGO_COLORS = {"simple_ga": "#1f77b4", "open_es": "#ff7f0e", "cma_es": "#2ca02c", "sep_cma_es": "#d62728"}
# task -> (label, weight dim, filename template). HalfCheetah runs have no task prefix.
TASKS = {
    "cartpole":    ("CartPole (d=80)", 80, "cartpole_{algo}_seed{seed}.npz"),
    "halfcheetah": ("HalfCheetah (d=390)", 390, "{algo}_seed{seed}.npz"),
}
SEED = 42
from paths import RUNS_DIR as BRAX_RUNS


def load(task, algo, dim, tpl):
    d = np.load(BRAX_RUNS / tpl.format(algo=algo, seed=SEED), allow_pickle=True)
    return d["populations"].astype(np.float64).reshape(-1, dim)


def joint_umap(task, dim, tpl):
    p = CACHE_DIR / f"exp30_joint_{task}_{SEED}.npz"
    Ws, names = [], []
    for algo in ALGOS:
        W = load(task, algo, dim, tpl); Ws.append(W); names += [algo] * len(W)
    names = np.array(names)
    if p.exists():
        return np.load(p)["emb"], names
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                    metric="euclidean", random_state=SEED).fit_transform(np.vstack(Ws))
    np.savez_compressed(p, emb=emb)
    return emb, names


def main():
    ensure_dirs()
    set_science_style()

    rows = []
    fig, axes = plt.subplots(1, len(TASKS), figsize=(4.4 * len(TASKS), 4.6))
    for ax, (task, (label, dim, tpl)) in zip(np.atleast_1d(axes), TASKS.items()):
        emb, names = joint_umap(task, dim, tpl)
        ax.set_facecolor("white")
        for algo in ALGOS:
            m = names == algo
            ax.scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.45, edgecolors="none",
                       color=ALGO_COLORS[algo], label=ALGO_LABELS[algo], rasterized=True)
            e = emb[m]
            rows.append({"task": label, "algorithm": ALGO_LABELS[algo],
                         "umap_clusters": count_attractors_dbscan(e, min_samples=5),
                         "compactness": round(float(np.linalg.norm(e - e.mean(0), axis=1).mean()), 3)})
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(label, fontsize=11)
    np.atleast_1d(axes)[0].legend(markerscale=2, fontsize=7, loc="best")
    save_results_csv(rows, RESULTS_DIR / "exp30_control_task_signatures.csv")

    print("Control-task signatures (seed 42):")
    print(f"  {'task':18s} {'algorithm':12s} {'clusters':>9s} {'compactness':>12s}")
    for r in rows:
        print(f"  {r['task']:18s} {r['algorithm']:12s} {r['umap_clusters']:9d} {r['compactness']:12.3f}")
    # consistency: GA most fragmented in each control task?
    print("\nMost-fragmented algorithm per control task (expect Simple GA):")
    for task, (label, *_rest) in TASKS.items():
        sub = [r for r in rows if r["task"] == label]
        top = max(sub, key=lambda r: r["umap_clusters"])
        print(f"  {label:18s}: {top['algorithm']} ({top['umap_clusters']} clusters)")

    fig.suptitle("ES search signatures on control tasks (joint UMAP, seed 42) — "
                 "GA fragments, distribution-ES stay compact", fontsize=11, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "exp30_control_task_signatures.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
