"""Exp 11: evolution summary for the benchmarks where neuroevolution worked.

CIFAR-10 barely learned (final fitness below the random baseline), so this figure
focuses on make_moons and halfcheetah. Top row: fitness trajectory (mean ± std
across seeds). Bottom row: the UMAP projection (euclidean) colored by generation.
Reads Exp 1 caches.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.shared import CACHE_DIR, FIGURES_DIR, ensure_dirs

BENCHMARKS = ["make_moons", "halfcheetah"]
SEEDS = [42, 123, 7, 31, 99]
PROJ_SEED = 42
DISPLAY = {"make_moons": "Make Moons", "halfcheetah": "HalfCheetah"}
RANDOM_BASELINE = {"make_moons": -np.log(2)}  # BCE chance level; halfcheetah reward has none


def load(benchmark: str, seed: int) -> dict | None:
    p = CACHE_DIR / f"exp1_{benchmark}_{seed}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return {
        "mean_fitness": d["mean_fitness"],
        "emb_all": d["emb_all"],
        "gen_labels": d["gen_labels"],
    }


def main() -> None:
    ensure_dirs()
    fig, axes = plt.subplots(2, len(BENCHMARKS), figsize=(6.2 * len(BENCHMARKS), 9), dpi=130)

    for c, benchmark in enumerate(BENCHMARKS):
        # --- top: fitness trajectory mean ± std across seeds ---
        ax_f = axes[0, c]
        curves = [load(benchmark, s)["mean_fitness"] for s in SEEDS if load(benchmark, s) is not None]
        if curves:
            L = min(len(x) for x in curves)
            M = np.vstack([x[:L] for x in curves])
            gens = np.arange(L)
            mean, std = M.mean(0), M.std(0)
            ax_f.plot(gens, mean, color="#1f77b4", lw=2, label="mean fitness")
            ax_f.fill_between(gens, mean - std, mean + std, color="#1f77b4", alpha=0.2, label="±1 std")
            if benchmark in RANDOM_BASELINE:
                ax_f.axhline(RANDOM_BASELINE[benchmark], ls="--", color="#d62728", lw=1.3,
                             label=f"random baseline ({RANDOM_BASELINE[benchmark]:.2f})")
            ax_f.set_xlabel("generation")
            ax_f.set_ylabel("fitness")
            ax_f.set_title(f"{DISPLAY[benchmark]} — fitness ({len(curves)} seeds)", fontsize=11)
            ax_f.legend(fontsize=8.5, loc="lower right")
            ax_f.grid(alpha=0.3)

        # --- bottom: UMAP projection (euclidean) colored by generation ---
        ax_p = axes[1, c]
        payload = load(benchmark, PROJ_SEED)
        ax_p.set_facecolor("white")
        if payload is not None:
            emb, gl = payload["emb_all"], payload["gen_labels"]
            n_gens = int(gl.max()) + 1 if len(gl) else 1
            sc = ax_p.scatter(emb[:, 0], emb[:, 1], c=gl, cmap="plasma", vmin=0, vmax=max(n_gens - 1, 1),
                              s=6, alpha=0.6, edgecolors="none", rasterized=True)
            fig.colorbar(sc, ax=ax_p, label="generation", shrink=0.8, pad=0.02)
            ax_p.set_title(f"{DISPLAY[benchmark]} — UMAP projection (euclidean, seed {PROJ_SEED})", fontsize=11)
        ax_p.set_xticks([]); ax_p.set_yticks([])

    fig.suptitle("Neuroevolution on benchmarks where it succeeded", fontsize=14, y=1.0)
    fig.tight_layout()
    out = FIGURES_DIR / "exp11_evolution_summary.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
