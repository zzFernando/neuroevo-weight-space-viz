"""Exp 21: far in weight space, close in function — empirical proof.

Takes the best individual found by each of the 4 ES algorithms on make_moons
(seed 42), which the joint UMAP placed in distant regions, and checks whether they
nevertheless compute the *same function*. Measures, pairwise:
  - weight-space Euclidean distance (expected: large)
  - correlation of raw outputs on the dataset (expected: ~1)
  - label agreement (expected: ~100%)
and draws the four decision boundaries side by side.

Confirms the claim that NN symmetries make functionally-equivalent solutions land
far apart in weight space — the reason the algorithms don't converge to one point.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


from shared import FIGURES_DIR, RESULTS_DIR, ensure_dirs, set_science_style, save_results_csv

ALGOS = ["simple_ga", "open_es", "cma_es", "sep_cma_es"]
ALGO_LABELS = {"simple_ga": "Simple GA", "open_es": "OpenES",
               "cma_es": "CMA-ES", "sep_cma_es": "sep-CMA-ES"}
SEED = 42
HIDDEN = 16
from paths import RUNS_DIR as BRAX_RUNS


def forward(flat, X):
    W1 = flat[:2 * HIDDEN].reshape(2, HIDDEN)
    W2 = flat[2 * HIDDEN:].reshape(HIDDEN, 1)
    h = np.tanh(X @ W1)
    return 1.0 / (1.0 + np.exp(-(h @ W2)[:, 0]))  # sigmoid prob


def best_individuals():
    best = {}
    for algo in ALGOS:
        d = np.load(BRAX_RUNS / f"moons_{algo}_seed{SEED}.npz", allow_pickle=True)
        pops = d["populations"].reshape(-1, 2 * HIDDEN + HIDDEN)
        fits = d["fitnesses"].reshape(-1)
        best[algo] = pops[int(np.argmax(fits))].astype(np.float64)
    return best


def main():
    ensure_dirs()
    set_science_style()

    X, y = make_moons(n_samples=1000, noise=0.25, random_state=SEED)
    X = StandardScaler().fit_transform(X)

    best = best_individuals()
    preds = {a: forward(w, X) for a, w in best.items()}
    labels = {a: (preds[a] > 0.5).astype(int) for a in ALGOS}

    # pairwise stats
    rows = []
    for i, a in enumerate(ALGOS):
        for b in ALGOS[i + 1:]:
            wdist = float(np.linalg.norm(best[a] - best[b]))
            ocorr = float(np.corrcoef(preds[a], preds[b])[0, 1])
            agree = float((labels[a] == labels[b]).mean())
            rows.append({"pair": f"{ALGO_LABELS[a]} vs {ALGO_LABELS[b]}",
                         "weight_distance": round(wdist, 3),
                         "output_correlation": round(ocorr, 4),
                         "label_agreement": round(agree, 4)})
    save_results_csv(rows, RESULTS_DIR / "exp21_functional_equivalence.csv")

    print("Pairwise: distant weights vs identical function?")
    print(f"  {'pair':28s} {'w_dist':>8s} {'out_corr':>9s} {'agree':>7s}")
    for r in rows:
        print(f"  {r['pair']:28s} {r['weight_distance']:8.2f} "
              f"{r['output_correlation']:9.4f} {r['label_agreement']*100:6.1f}%")

    # decision boundaries side by side
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .5, X[:, 0].max() + .5, 300),
                         np.linspace(X[:, 1].min() - .5, X[:, 1].max() + .5, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, len(ALGOS), figsize=(4.0 * len(ALGOS), 4.2))
    for ax, algo in zip(axes, ALGOS):
        zz = forward(best[algo], grid).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.7)
        ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.2)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", s=4, edgecolors="k", linewidths=0.15)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ALGO_LABELS[algo], fontsize=10)

    mean_corr = np.mean([r["output_correlation"] for r in rows])
    mean_wdist = np.mean([r["weight_distance"] for r in rows])
    fig.suptitle(f"Same decision boundary, distant weights — make_moons best individuals\n"
                 f"(mean pairwise weight-dist={mean_wdist:.1f}, mean output-corr={mean_corr:.3f})",
                 fontsize=12, y=1.04)
    out = FIGURES_DIR / "exp21_functional_equivalence.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
