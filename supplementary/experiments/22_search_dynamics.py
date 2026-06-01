"""Exp 22: quantitative search-dynamics curves per generation.

Turns the visual geometry diagnostics into objective time-series, computed in the
RAW weight space (not the UMAP embedding, which distorts distances). For each
algorithm/generation:
  - spread          : mean ||w_i - centroid||  (population dispersion; contraction = convergence)
  - centroid_velocity: ||centroid_g - centroid_{g-1}||  (how fast the search moves)
  - mean_fitness    : reference progress curve
Plotted for the 4 ES algorithms on both benchmarks (seed 42).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.shared import FIGURES_DIR, RESULTS_DIR, ensure_dirs, set_science_style, save_results_csv

ALGOS = ["simple_ga", "open_es", "cma_es", "sep_cma_es"]
ALGO_LABELS = {"simple_ga": "Simple GA", "open_es": "OpenES",
               "cma_es": "CMA-ES", "sep_cma_es": "sep-CMA-ES"}
ALGO_COLORS = {"simple_ga": "#1f77b4", "open_es": "#ff7f0e",
               "cma_es": "#2ca02c", "sep_cma_es": "#d62728"}
SEED = 42
BENCHMARKS = {"make_moons": "moons_{algo}_seed42.npz", "halfcheetah": "{algo}_seed42.npz"}
BRAX_RUNS = ROOT.parent / "brax" / "runs"


def dynamics(pops: np.ndarray, fits: np.ndarray):
    """pops (G,P,D), fits (G,P) -> per-gen spread, centroid velocity, mean fitness."""
    centroids = pops.mean(axis=1)                                  # (G, D)
    spread = np.linalg.norm(pops - centroids[:, None, :], axis=2).mean(axis=1)  # (G,)
    vel = np.r_[0.0, np.linalg.norm(np.diff(centroids, axis=0), axis=1)]        # (G,)
    return spread, vel, fits.mean(axis=1)


def main():
    ensure_dirs()
    set_science_style()

    metrics = [("spread", "population spread\n(mean dist. to centroid)"),
               ("velocity", "centroid velocity\n(per generation)"),
               ("fitness", "mean fitness")]
    fig, axes = plt.subplots(len(metrics), len(BENCHMARKS),
                             figsize=(6.5 * len(BENCHMARKS), 4.0 * len(metrics)))
    rows_csv = []

    for col, (bench, tpl) in enumerate(BENCHMARKS.items()):
        for algo in ALGOS:
            f = BRAX_RUNS / tpl.format(algo=algo)
            if not f.exists():
                continue
            d = np.load(f, allow_pickle=True)
            spread, vel, mfit = dynamics(d["populations"].astype(np.float64),
                                         d["fitnesses"].astype(float))
            gens = np.arange(len(spread))
            for (key, _), series in zip(metrics, (spread, vel, mfit)):
                ax = axes[[m[0] for m in metrics].index(key), col]
                ax.plot(gens, series, color=ALGO_COLORS[algo], lw=1.6, label=ALGO_LABELS[algo])
            for g, (s, v, mf) in enumerate(zip(spread, vel, mfit)):
                rows_csv.append({"benchmark": bench, "algorithm": ALGO_LABELS[algo],
                                 "generation": g, "spread": round(float(s), 4),
                                 "centroid_velocity": round(float(v), 4),
                                 "mean_fitness": round(float(mf), 4)})

        for r, (key, ylabel) in enumerate(metrics):
            ax = axes[r, col]
            if r == 0:
                ax.set_title(bench, fontsize=11)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if r == len(metrics) - 1:
                ax.set_xlabel("generation", fontsize=9)
            ax.grid(alpha=0.3)
            if r == 0 and col == 0:
                ax.legend(fontsize=8, loc="best")

    save_results_csv(rows_csv, RESULTS_DIR / "exp22_search_dynamics.csv")
    fig.suptitle(f"Search dynamics in raw weight space — 4 ES algorithms (seed {SEED})",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    out = FIGURES_DIR / "exp22_search_dynamics.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")

    # quick numeric summary: contraction (spread end/start) & total drift
    print("\nfinal/initial spread ratio (↓ = converged) and total centroid drift:")
    for bench, tpl in BENCHMARKS.items():
        print(f"  {bench}:")
        for algo in ALGOS:
            f = BRAX_RUNS / tpl.format(algo=algo)
            if not f.exists():
                continue
            d = np.load(f, allow_pickle=True)
            spread, vel, _ = dynamics(d["populations"].astype(np.float64), d["fitnesses"].astype(float))
            ratio = spread[-1] / (spread[0] + 1e-9)
            print(f"    {ALGO_LABELS[algo]:12s}  spread_ratio={ratio:5.2f}  total_drift={vel.sum():8.2f}")


if __name__ == "__main__":
    main()
