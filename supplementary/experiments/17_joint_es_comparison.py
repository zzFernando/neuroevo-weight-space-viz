"""Exp 17: joint UMAP of multiple neuroevolution algorithms in a shared weight space.

The 4 ES algorithms (simple_ga, open_es, cma_es, sep_cma_es) all optimize the same
390-dim HalfCheetah policy, so their populations live in ONE comparable space. We fit
a single UMAP on all of them pooled — separate per-algorithm fits would not be
comparable (each UMAP has an arbitrary rotation/scale). Colored by algorithm we see
how each ES covers the weight space; colored by fitness we see where good solutions lie.

Outputs:
  exp17_joint_es_by_algorithm.png   — single panel (option 1)
  exp17_joint_es_algo_vs_fitness.png — two panels: algorithm | fitness (option 2)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.shared import CACHE_DIR, FIGURES_DIR, ensure_dirs, set_science_style

SEED = 42
ALGOS = ["simple_ga", "open_es", "cma_es", "sep_cma_es"]
ALGO_LABELS = {"simple_ga": "Simple GA", "open_es": "OpenES",
               "cma_es": "CMA-ES", "sep_cma_es": "sep-CMA-ES"}
ALGO_COLORS = {"simple_ga": "#1f77b4", "open_es": "#ff7f0e",
               "cma_es": "#2ca02c", "sep_cma_es": "#d62728"}
BRAX_RUNS = ROOT.parent / "brax" / "runs"


def load_all():
    """Pool every algorithm's populations into one matrix, tracking labels."""
    W, algo_idx, gen_idx, fit = [], [], [], []
    for ai, algo in enumerate(ALGOS):
        d = np.load(BRAX_RUNS / f"{algo}_seed{SEED}.npz", allow_pickle=True)
        pops = d["populations"].astype(np.float64)   # (G, P, 390)
        f = d["fitnesses"].astype(float)             # (G, P)
        G, P, _ = pops.shape
        W.append(pops.reshape(G * P, -1))
        algo_idx.append(np.full(G * P, ai))
        gen_idx.append(np.repeat(np.arange(G), P))
        fit.append(f.reshape(-1))
    return (np.vstack(W), np.concatenate(algo_idx),
            np.concatenate(gen_idx), np.concatenate(fit))


def joint_embedding(W: np.ndarray) -> np.ndarray:
    p = CACHE_DIR / f"exp17_joint_emb_{SEED}.npz"
    if p.exists():
        return np.load(p)["emb"]
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                        metric="euclidean", random_state=SEED)
    emb = reducer.fit_transform(W)
    np.savez_compressed(p, emb=emb)
    return emb


def scatter_by_algorithm(ax, emb, algo_idx):
    ax.set_facecolor("white")
    for ai, algo in enumerate(ALGOS):
        m = algo_idx == ai
        ax.scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.45, edgecolors="none",
                   color=ALGO_COLORS[algo], label=ALGO_LABELS[algo], rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    leg = ax.legend(markerscale=4, fontsize=8, loc="best")
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)


def scatter_by_fitness(ax, emb, fit):
    ax.set_facecolor("white")
    fmin, fmax = np.percentile(fit, 50), np.percentile(fit, 99)
    c = np.clip((fit - fmin) / (fmax - fmin + 1e-9) * 100, 0, 100)
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=c, cmap="viridis", s=3, alpha=0.55,
                    edgecolors="none", rasterized=True)
    best = int(np.argmax(fit))
    ax.scatter(emb[best, 0], emb[best, 1], marker="*", s=160, c="#d62728",
               edgecolors="white", linewidths=0.8, zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    return sc


def main():
    ensure_dirs()
    set_science_style()
    W, algo_idx, gen_idx, fit = load_all()
    print(f"Pooled {W.shape[0]} individuals × {W.shape[1]} dims from {len(ALGOS)} algorithms")
    emb = joint_embedding(W)

    # --- Option 1: single panel, by algorithm ---
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_by_algorithm(ax, emb, algo_idx)
    ax.set_title(f"Joint UMAP of neuroevolution algorithms — shared 390-dim weight space\n"
                 f"(HalfCheetah, seed {SEED}, euclidean)", fontsize=10)
    out1 = FIGURES_DIR / "exp17_joint_es_by_algorithm.png"
    fig.savefig(out1, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out1}")

    # --- Option 2: two panels, algorithm | fitness ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    scatter_by_algorithm(axes[0], emb, algo_idx)
    axes[0].set_title("colored by algorithm", fontsize=10)
    sc = scatter_by_fitness(axes[1], emb, fit)
    axes[1].set_title("colored by fitness  (★ = global best)", fontsize=10)
    fig.colorbar(sc, ax=axes[1], label="fitness percentile", shrink=0.75, pad=0.02)
    fig.suptitle(f"Joint UMAP of 4 ES algorithms in a shared weight space "
                 f"(HalfCheetah, seed {SEED})", fontsize=12, y=1.0)
    out2 = FIGURES_DIR / "exp17_joint_es_algo_vs_fitness.png"
    fig.savefig(out2, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out2}")

    # quick quantitative: weight-space coverage (embedding bbox area) per algorithm
    print("\nweight-space coverage (UMAP bbox area) & best fitness per algorithm:")
    for ai, algo in enumerate(ALGOS):
        m = algo_idx == ai
        e = emb[m]
        area = (e[:, 0].max() - e[:, 0].min()) * (e[:, 1].max() - e[:, 1].min())
        print(f"  {ALGO_LABELS[algo]:12s}  coverage={area:8.1f}   best_fit={fit[m].max():7.1f}")


if __name__ == "__main__":
    main()
