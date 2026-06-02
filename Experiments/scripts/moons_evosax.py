"""Neuroevolution on make_moons via evosax strategies — mirrors neuroevolve_evosax.py.

Lets us run the SAME multi-ES joint-UMAP comparison on make_moons as on HalfCheetah.
Architecture matches supplementary/benchmarks/moons.py exactly:
    X(2) -> tanh -> hidden(16) -> sigmoid(1)   (no biases)   d_w = 2*16 + 16*1 = 48
Fitness = -binary cross-entropy (maximize). The objective is deterministic (fixed
dataset), so no per-individual rollout keys are needed.

Saves runs/moons_{algo}_seed{seed}.npz with the same keys as the brax runs
('populations' G×P×D, 'fitnesses' G×P) so experiment 19 can consume them directly.

Usage: .venv/bin/python moons_evosax.py --all_algos --seed 42
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HIDDEN = 16
N_PARAMS = 2 * HIDDEN + HIDDEN * 1  # 48


def _load_strategies():
    from evosax.algorithms import CMA_ES, Sep_CMA_ES
    from evosax.algorithms.distribution_based.open_es import Open_ES
    from evosax.algorithms.population_based.simple_ga import SimpleGA
    return {"simple_ga": SimpleGA, "open_es": Open_ES,
            "cma_es": CMA_ES, "sep_cma_es": Sep_CMA_ES}


def make_data(seed: int, task: str = "moons"):
    # Datasets are pre-generated with sklearn in the supplementary env (sklearn is not
    # installed here): runs/{task}_data_seed{seed}.npz with keys X, y. All are 2-D binary.
    d = np.load(Path("runs") / f"{task}_data_seed{seed}.npz")
    return jnp.asarray(d["X"], jnp.float32), jnp.asarray(d["y"], jnp.float32)


def build_loss_fn(X, y):
    """Returns per-individual BCE loss (to MINIMIZE — matches evosax convention)."""
    def single(flat):
        W1 = flat[:2 * HIDDEN].reshape(2, HIDDEN)
        W2 = flat[2 * HIDDEN:].reshape(HIDDEN, 1)
        h = jnp.tanh(X @ W1)
        p = jax.nn.sigmoid((h @ W2)[:, 0])
        eps = 1e-7
        return -(y * jnp.log(p + eps) + (1 - y) * jnp.log(1 - p + eps)).mean()  # BCE
    return jax.jit(jax.vmap(single))


def run_evosax(algo, loss_fn, pop_size, n_gens, seed, log_every=20):
    StrategyCls = _load_strategies()[algo]
    strategy = StrategyCls(population_size=pop_size, solution=jnp.zeros(N_PARAMS))
    params = strategy.default_params

    rng = jax.random.PRNGKey(seed)
    rng, ri = jax.random.split(rng)
    try:
        state = strategy.init(ri, jnp.zeros(N_PARAMS), params)
    except TypeError:
        init_pop = jax.random.normal(ri, (pop_size, N_PARAMS)) * 0.5
        rng, ri2 = jax.random.split(rng)
        state = strategy.init(ri2, init_pop, jnp.zeros(pop_size), params)

    hist_pop = np.zeros((n_gens, pop_size, N_PARAMS), dtype=np.float32)
    hist_fit = np.zeros((n_gens, pop_size), dtype=np.float32)  # stored as fitness = -loss (higher better)
    t0 = time.time()
    for g in range(n_gens):
        rng, r_ask, r_tell = jax.random.split(rng, 3)
        pop, state = strategy.ask(r_ask, state, params)
        loss = np.asarray(loss_fn(pop))                       # BCE, to MINIMIZE
        state, _ = strategy.tell(r_tell, pop, jnp.array(loss), state, params)  # evosax minimizes
        hist_pop[g] = np.asarray(pop)
        hist_fit[g] = -loss                                   # save as fitness (higher = better)
        if g % log_every == 0 or g == n_gens - 1:
            print(f"  [{algo}] gen {g:3d}  best_fit={-loss.min():.3f}  mean_fit={-loss.mean():.3f}  ({time.time()-t0:.1f}s)")
    return hist_pop, hist_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="open_es")
    ap.add_argument("--all_algos", action="store_true")
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--gens", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task", default="moons", help="moons|circles|blobs|xor (2-D binary)")
    ap.add_argument("--out_dir", type=Path, default=Path("runs"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    X, y = make_data(args.seed, args.task)
    loss_fn = build_loss_fn(X, y)
    algos = list(_load_strategies()) if args.all_algos else [args.algo]

    for algo in algos:
        print(f"=== {args.task}/{algo} (seed {args.seed}) ===")
        pops, fits = run_evosax(algo, loss_fn, args.pop, args.gens, args.seed)
        out = args.out_dir / f"{args.task}_{algo}_seed{args.seed}.npz"
        np.savez_compressed(out, populations=pops, fitnesses=fits)
        print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
