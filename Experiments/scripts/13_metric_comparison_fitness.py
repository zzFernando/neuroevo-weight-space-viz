"""Exp 13: same metric comparison as Exp 12, but colored by fitness.

Reuses the Exp 12 cached UMAP embeddings (make_moons + halfcheetah, the benchmarks
where evolution worked) and recolors each projection by per-individual fitness
(viridis, larger = fitter, red star = best), instead of by generation.

Fitness sources:
  make_moons  — recomputed by evaluating each cached (flattened) weight vector.
  halfcheetah — read from brax/runs/simple_ga_seed{seed}.npz key 'fitnesses'.
The flattened ordering (generation-major) matches how compute_aligned_umap_embedding
stacks emb_all, so fitness aligns row-for-row with the embedding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


from benchmarks.moons import NeuroEvoMoons
from benchmarks import DEFAULTS
from shared import CACHE_DIR, FIGURES_DIR, ensure_dirs, plot_fitness_panel, set_science_style

METRICS = ["euclidean", "mahalanobis", "chebyshev", "cosine", "correlation"]
SEED = 42
DISPLAY = {"make_moons": "Make Moons (low-dim)", "halfcheetah": "HalfCheetah (390-dim)"}
from paths import RUNS_DIR as BRAX_RUNS


def make_moons_fitness(seed: int) -> np.ndarray | None:
    """Reconstruct per-individual fitness (generation-major) from cached weights."""
    for p in (CACHE_DIR / f"exp9_evo_make_moons_{seed}.npz",
              CACHE_DIR / f"exp10_evo_make_moons_{seed}.npz"):
        if p.exists():
            wbg = list(np.load(p, allow_pickle=True)["weights_by_gen"])
            break
    else:
        return None

    cfg = DEFAULTS["make_moons"]
    env = NeuroEvoMoons(pop_size=cfg["pop_size"], hidden_dim=cfg["hidden_dim"],
                        mutation_rate=cfg["mutation_rate"], seed=seed)
    (s1, s2) = env.shapes
    n1 = int(np.prod(s1))

    fits = []
    for gen in wbg:
        for flat in np.asarray(gen, dtype=np.float64):
            w1 = flat[:n1].reshape(s1)
            w2 = flat[n1:].reshape(s2)
            fits.append(env.evaluate([w1, w2]))
    return np.asarray(fits, dtype=float)


def halfcheetah_fitness(seed: int) -> np.ndarray | None:
    p = BRAX_RUNS / f"simple_ga_seed{seed}.npz"
    if not p.exists():
        return None
    return np.asarray(np.load(p, allow_pickle=True)["fitnesses"], dtype=float).ravel()


def load_embedding(benchmark: str, metric: str) -> dict | None:
    p = CACHE_DIR / f"exp12_emb_{benchmark}_{SEED}_{metric}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return {"emb_all": d["emb_all"], "gen_labels": d["gen_labels"]}


def main() -> None:
    ensure_dirs()
    set_science_style()
    benchmarks = ["make_moons", "halfcheetah"]
    fitness = {"make_moons": make_moons_fitness(SEED), "halfcheetah": halfcheetah_fitness(SEED)}

    nrows, ncols = len(benchmarks), len(METRICS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.5 * nrows), dpi=130)
    axes = np.atleast_2d(axes)
    last_sc = None

    for r, b in enumerate(benchmarks):
        fit = fitness[b]
        for c, m in enumerate(METRICS):
            ax = axes[r, c]
            ax.set_facecolor("white")
            emb_payload = load_embedding(b, m)
            if emb_payload is None or fit is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", color="#a00")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                emb = emb_payload["emb_all"]
                f = fit
                if len(f) != len(emb):  # guard against any ordering/length mismatch
                    n = min(len(f), len(emb))
                    emb, f = emb[:n], f[:n]
                last_sc = plot_fitness_panel(ax, emb, f, title=m)
            if c == 0:
                ax.set_ylabel(DISPLAY[b], fontsize=10)

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes[:, -1], orientation="vertical",
                            label="fitness percentile", shrink=0.6, pad=0.02)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"UMAP metric comparison colored by fitness — make_moons & halfcheetah  "
        f"(nn=15, min_dist=0.1, λ=0.8, seed={SEED}; ★ = best individual)", fontsize=12, y=1.01)
    fig.tight_layout()
    out = FIGURES_DIR / "exp13_metric_comparison_fitness.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
