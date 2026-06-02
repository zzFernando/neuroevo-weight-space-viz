"""Exp 27: offspring allocation guided by fitness localization.

Demonstrates a second application the paper only *proposed*: using the spatial
localization of high-fitness individuals to guide non-uniform offspring allocation.
We compare three parent-selection strategies in an otherwise identical GA on make_moons:

  - uniform           : each elite gets equal expected offspring (baseline / paper default)
  - fitness_weighted  : offspring ∝ softmax(rank) — concentrate around the fittest (attractor)
  - diversity_weighted: offspring ∝ distance to elite centroid — favor spread-out elites

Honest framing (ties to Exp 26): concentrating too hard risks premature convergence;
the useful regime balances exploitation of high-fitness regions with diversity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import DEFAULTS
from benchmarks.moons import NeuroEvoMoons
from experiments.shared import FIGURES_DIR, RESULTS_DIR, ensure_dirs, set_science_style, save_results_csv

STRATEGIES = ["uniform", "fitness_weighted", "diversity_weighted"]
LABELS = {"uniform": "Uniform (baseline)", "fitness_weighted": "Fitness-weighted",
          "diversity_weighted": "Diversity-weighted"}
COLORS = {"uniform": "#1f77b4", "fitness_weighted": "#d62728", "diversity_weighted": "#2ca02c"}
SEEDS = [42, 7, 123, 31, 99]
N_GENS = 80


def parent_probs(strategy, elite_fitness, elites_flat):
    n = len(elite_fitness)
    if strategy == "uniform":
        return np.full(n, 1.0 / n)
    if strategy == "fitness_weighted":
        ranks = np.argsort(np.argsort(elite_fitness))  # 0..n-1, higher fitness → higher rank
        w = np.exp(1.5 * ranks / max(n - 1, 1))         # softmax over rank (concentrate on best)
        return w / w.sum()
    if strategy == "diversity_weighted":
        centroid = elites_flat.mean(0)
        d = np.linalg.norm(elites_flat - centroid, axis=1)
        w = d + 1e-6
        return w / w.sum()
    raise ValueError(strategy)


def run(strategy, seed):
    cfg = DEFAULTS["make_moons"]
    env = NeuroEvoMoons(pop_size=cfg["pop_size"], hidden_dim=cfg["hidden_dim"],
                        mutation_rate=cfg["mutation_rate"], seed=seed)
    n_elite = max(2, int(env.pop_size * 0.2))
    mean_fit, diversity = [], []
    for _ in range(N_GENS):
        fitness = np.array([env.evaluate(ind) for ind in env.population])
        flat = np.array([env.flatten(ind) for ind in env.population])
        mean_fit.append(float(fitness.mean()))
        diversity.append(float(np.linalg.norm(flat - flat.mean(0), axis=1).mean()))

        elite_idx = np.argsort(fitness)[-n_elite:]
        elites = [env.population[i] for i in elite_idx]
        elites_flat = flat[elite_idx]
        probs = parent_probs(strategy, fitness[elite_idx], elites_flat)

        new_pop = elites.copy()
        while len(new_pop) < env.pop_size:
            parent = elites[env.rng.choice(n_elite, p=probs)]
            new_pop.append(env.mutate(parent))
        env.population = new_pop
    return np.array(mean_fit), np.array(diversity)


def main():
    ensure_dirs()
    set_science_style()

    curves = {s: [] for s in STRATEGIES}
    divs = {s: [] for s in STRATEGIES}
    rows = []
    for strategy in STRATEGIES:
        finals = []
        for seed in SEEDS:
            mf, dv = run(strategy, seed)
            curves[strategy].append(mf); divs[strategy].append(dv)
            finals.append(mf[-1])
        finals = np.array(finals)
        rows.append({"strategy": LABELS[strategy],
                     "final_fitness_mean": round(float(finals.mean()), 4),
                     "final_fitness_std": round(float(finals.std()), 4)})
    save_results_csv(rows, RESULTS_DIR / "exp27_offspring_allocation.csv")
    print("strategy              final_fitness (mean ± std, 5 seeds)")
    for r in rows:
        print(f"  {r['strategy']:22s} {r['final_fitness_mean']:+.4f} ± {r['final_fitness_std']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for strategy in STRATEGIES:
        C = np.vstack(curves[strategy]); D = np.vstack(divs[strategy])
        g = np.arange(C.shape[1])
        m, s = C.mean(0), C.std(0)
        axes[0].plot(g, m, color=COLORS[strategy], lw=1.8, label=LABELS[strategy])
        axes[0].fill_between(g, m - s, m + s, color=COLORS[strategy], alpha=0.15)
        axes[1].plot(g, D.mean(0), color=COLORS[strategy], lw=1.8, label=LABELS[strategy])
    axes[0].set_title("mean fitness", fontsize=11); axes[0].set_xlabel("generation")
    axes[0].legend(fontsize=8, title="offspring allocation"); axes[0].grid(alpha=0.3)
    axes[1].set_title("population diversity", fontsize=11); axes[1].set_xlabel("generation")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Offspring allocation guided by fitness localization (make_moons, 5 seeds)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "exp27_offspring_allocation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
