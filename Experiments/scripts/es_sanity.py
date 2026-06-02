"""Sanity-check: do the evosax ES implementations actually optimize correctly?

Runs the 4 strategies on synthetic functions with a KNOWN global optimum (0):
  - sphere     : convex, unimodal       — every decent ES should reach ~0
  - rastrigin  : highly multimodal       — separates strong (CMA) from weak (GA) optimizers
This validates that the "search signatures" seen in the UMAP study are real algorithm
behavior, not implementation artifacts. Saves a convergence figure + CSV.

Usage: .venv/bin/python es_sanity.py
"""
from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from paths import RUNS_DIR, FIGURES_DIR, RESULTS_DIR

DIM = 16
POP = 50
GENS = 100
SEEDS = [42, 7, 123]


def _load_strategies():
    from evosax.algorithms import CMA_ES, Sep_CMA_ES
    from evosax.algorithms.distribution_based.open_es import Open_ES
    from evosax.algorithms.population_based.simple_ga import SimpleGA
    return {"simple_ga": SimpleGA, "open_es": Open_ES, "cma_es": CMA_ES, "sep_cma_es": Sep_CMA_ES}


def sphere(x):                      # optimum 0 at x=0
    return jnp.sum(x ** 2, axis=-1)


def rastrigin(x):                   # optimum 0 at x=0
    return 10 * x.shape[-1] + jnp.sum(x ** 2 - 10 * jnp.cos(2 * jnp.pi * x), axis=-1)


OBJ = {"sphere": sphere, "rastrigin": rastrigin}


def run(algo, obj, seed):
    StrategyCls = _load_strategies()[algo]
    strat = StrategyCls(population_size=POP, solution=jnp.zeros(DIM))
    params = strat.default_params

    rng = jax.random.PRNGKey(seed)
    rng, ri = jax.random.split(rng)
    try:
        state = strat.init(ri, jnp.zeros(DIM), params)
    except TypeError:
        rng, ri2 = jax.random.split(rng)
        state = strat.init(ri2, jax.random.normal(ri, (POP, DIM)) * 0.5, jnp.zeros(POP), params)

    obj_jit = jax.jit(obj)
    best_curve = []
    for _ in range(GENS):
        rng, ra, rt = jax.random.split(rng, 3)
        pop, state = strat.ask(ra, state, params)
        f = np.asarray(obj_jit(pop))           # objective to MINIMIZE
        state, _ = strat.tell(rt, pop, jnp.array(f), state, params)  # evosax minimizes
        best_curve.append(float(f.min()))
    return np.minimum.accumulate(best_curve)   # best-so-far


def main():
    import matplotlib.pyplot as plt
    try:
        import scienceplots  # noqa
        plt.style.use(["science", "no-latex", "grid"])
    except Exception:
        pass

    algos = list(_load_strategies())
    colors = {"simple_ga": "#1f77b4", "open_es": "#ff7f0e", "cma_es": "#2ca02c", "sep_cma_es": "#d62728"}
    labels = {"simple_ga": "Simple GA", "open_es": "OpenES", "cma_es": "CMA-ES", "sep_cma_es": "sep-CMA-ES"}

    fig, axes = plt.subplots(1, len(OBJ), figsize=(6.2 * len(OBJ), 4.4))
    rows = []
    t0 = time.time()
    for ax, (oname, obj) in zip(axes, OBJ.items()):
        for algo in algos:
            curves = np.vstack([run(algo, obj, s) for s in SEEDS])
            mean, std = curves.mean(0), curves.std(0)
            g = np.arange(GENS)
            ax.plot(g, mean, color=colors[algo], lw=1.7, label=labels[algo])
            ax.fill_between(g, mean - std, mean + std, color=colors[algo], alpha=0.15)
            rows.append({"function": oname, "algorithm": labels[algo],
                         "final_best": round(float(mean[-1]), 4), "optimum": 0.0})
            print(f"  {oname:10s} {labels[algo]:12s} final={mean[-1]:.4f} (opt=0)  [{time.time()-t0:.0f}s]")
        ax.set_yscale("log"); ax.set_xlabel("generation"); ax.set_ylabel("best objective (↓, opt=0)")
        ax.set_title(f"{oname} (dim={DIM})", fontsize=11); ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("ES sanity-check on functions with known optimum (mean ± std, 3 seeds)", fontsize=12, y=1.02)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figp = FIGURES_DIR / "exp25_es_sanity.png"
    fig.savefig(figp, dpi=200, bbox_inches="tight", facecolor="white")
    import csv
    with open(RESULTS_DIR / "exp25_es_sanity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"Saved {figp}")


if __name__ == "__main__":
    main()
