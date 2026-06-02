"""Exp 26: detecting premature convergence from population dynamics.

Demonstrates an application the paper only *proposed*: using population geometry as an
early-warning signal. We induce a spectrum of regimes by varying the mutation scale σ
on make_moons (low σ → premature collapse; healthy σ → sustained search), and track per
generation, in raw weight space:
  - mean fitness
  - population diversity (mean distance to centroid)  ← the velocity-field's scalar essence
  - centroid velocity (how far the distribution moves)

Claim: a diversity/velocity collapse while fitness is still poor flags premature
convergence, and it appears at or before the fitness plateau — actionable lead time
to trigger a diversity-restoring intervention.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


from benchmarks import DEFAULTS
from utils import run_evolution_benchmark
from shared import (
    CACHE_DIR, FIGURES_DIR, RESULTS_DIR, ensure_dirs, set_science_style, save_results_csv,
)

SIGMAS = [0.02, 0.05, 0.11, 0.20]
SEEDS = [42, 7, 123]
N_GENS = 80
POP = 50
HIDDEN = 16
COLORS = plt.cm.viridis(np.linspace(0.1, 0.9, len(SIGMAS)))


def run(sigma, seed):
    p = CACHE_DIR / f"exp26_moons_s{sigma}_seed{seed}.npz"
    if p.exists():
        d = np.load(p, allow_pickle=True)
        return list(d["weights_by_gen"]), d["mean_fitness"]
    res = run_evolution_benchmark(benchmark_name="make_moons", pop_size=POP, n_generations=N_GENS,
                                  hidden_dim=HIDDEN, mutation_rate=sigma, seed=seed)
    np.savez_compressed(p, weights_by_gen=np.array(res.weights_by_gen, dtype=object),
                        mean_fitness=res.mean_fitness)
    return res.weights_by_gen, res.mean_fitness


def dynamics(weights_by_gen):
    W = [np.asarray(g, dtype=np.float64) for g in weights_by_gen]
    centroids = np.array([g.mean(0) for g in W])
    diversity = np.array([np.linalg.norm(g - g.mean(0), axis=1).mean() for g in W])
    velocity = np.r_[0.0, np.linalg.norm(np.diff(centroids, axis=0), axis=1)]
    return diversity, velocity


def plateau_gen(mean_fitness, frac=0.95):
    f = np.asarray(mean_fitness, dtype=float)
    span = f[-1] - f[0]
    if abs(span) < 1e-9:
        return 1
    thr = f[0] + frac * span
    reached = np.where(f >= thr)[0]
    return int(reached[0]) if len(reached) else len(f) - 1


def collapse_gen(diversity, frac=0.10):
    """First gen where diversity drops below frac of its peak (population froze)."""
    d = np.asarray(diversity)
    thr = d.max() * frac + d.min() * (1 - frac)  # frac of the way down from peak
    below = np.where(d <= thr)[0]
    return int(below[0]) if len(below) else len(d) - 1


def main():
    ensure_dirs()
    set_science_style()

    curves = {}   # sigma -> (fit, div, vel) averaged over seeds
    rows = []
    points = []   # per (sigma, seed): early diversity vs final fitness
    for sigma in SIGMAS:
        fits, divs, vels, final_f, t_plat, t_coll = [], [], [], [], [], []
        for seed in SEEDS:
            w, mf = run(sigma, seed)
            div, vel = dynamics(w)
            L = min(len(mf), len(div))
            fits.append(np.asarray(mf)[:L]); divs.append(div[:L]); vels.append(vel[:L])
            final_f.append(float(mf[-1]))
            t_plat.append(plateau_gen(mf)); t_coll.append(collapse_gen(div))
            points.append((sigma, float(div[1:6].mean()), float(mf[-1])))  # early diversity (gen1-5)
        curves[sigma] = (np.mean(fits, 0), np.mean(divs, 0), np.mean(vels, 0))
        rows.append({"sigma": sigma,
                     "final_fitness": round(float(np.mean(final_f)), 3),
                     "early_diversity_gen1_5": round(float(np.mean([p[1] for p in points if p[0]==sigma])), 3),
                     "diversity_collapse_gen": round(float(np.mean(t_coll)), 1),
                     "fitness_plateau_gen": round(float(np.mean(t_plat)), 1),
                     "lead_time_gens": round(float(np.mean(t_plat)) - float(np.mean(t_coll)), 1)})
    save_results_csv(rows, RESULTS_DIR / "exp26_premature_convergence.csv")

    print("σ      final_fit   collapse_gen  plateau_gen  lead_time")
    for r in rows:
        print(f"  {r['sigma']:.2f}   {r['final_fitness']:8.3f}   {r['diversity_collapse_gen']:11.1f}  "
              f"{r['fitness_plateau_gen']:10.1f}  {r['lead_time_gens']:+.1f}")

    # ---- figure: fitness | diversity | velocity vs generation ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    titles = ["mean fitness", "population diversity\n(mean dist. to centroid)", "centroid velocity"]
    for k, (ax, title) in enumerate(zip(axes, titles)):
        for sigma, c in zip(SIGMAS, COLORS):
            series = curves[sigma][k]
            ax.plot(np.arange(len(series)), series, color=c, lw=1.8, label=f"σ={sigma}")
        ax.set_xlabel("generation"); ax.set_title(title, fontsize=11); ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8, title="mutation scale")
    # mark premature run: lowest sigma
    axes[0].axhline(curves[SIGMAS[-1]][0][-1], ls=":", color="#888", lw=1)
    fig.suptitle("Detecting premature convergence from population dynamics (make_moons, 3 seeds)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "exp26_premature_convergence.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")

    # ---- diagnostic: early diversity (gen 1-5) predicts final fitness ----
    fig, ax = plt.subplots(figsize=(6.6, 5))
    cmap = {s: c for s, c in zip(SIGMAS, COLORS)}
    for sigma, ediv, ff in points:
        ax.scatter(ediv, ff, s=90, color=cmap[sigma], edgecolors="k", linewidths=0.5, zorder=3)
    # legend by sigma
    for sigma in SIGMAS:
        ax.scatter([], [], color=cmap[sigma], s=90, edgecolors="k", linewidths=0.5, label=f"σ={sigma}")
    best = max(points, key=lambda p: p[2])
    ax.annotate("best regime", (best[1], best[2]), textcoords="offset points", xytext=(8, -2), fontsize=9)
    ax.set_xlabel("early population diversity (mean, gens 1–5)")
    ax.set_ylabel("final mean fitness (↑ better)")
    ax.set_title("Early diversity is a leading indicator: too low (frozen) or too high (chaotic)\n"
                 "both yield worse fitness — readable by generation 5", fontsize=10)
    ax.legend(fontsize=8, title="mutation scale"); ax.grid(alpha=0.3)
    fig.tight_layout()
    out2 = FIGURES_DIR / "exp26_premature_diagnostic.png"
    fig.savefig(out2, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
