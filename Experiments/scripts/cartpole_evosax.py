"""Neuroevolution on classic CartPole via evosax — a control task for cross-task signatures.

Self-contained JAX implementation of the classic CartPole dynamics (no brax env needed),
with the same evosax harness as neuroevolve_evosax.py. Policy: obs(4) → tanh(16) → 1
(action = output > 0), no biases → d_w = 4*16 + 16 = 80. Fitness = total steps balanced
(max 200). Deterministic init (fixed small perturbation) so fitness is a clean function of
weights. Saves runs/cartpole_{algo}_seed{seed}.npz (populations, fitnesses).

⚠️ evosax MINIMIZES tell(): we pass -reward to maximize balance time.

Usage: .venv/bin/python cartpole_evosax.py --all_algos --seed 42
"""
from __future__ import annotations

import argparse, time
from pathlib import Path

import jax, jax.numpy as jnp
import numpy as np

HIDDEN = 16
N_PARAMS = 4 * HIDDEN + HIDDEN  # 80
MAX_STEPS = 200

# classic CartPole constants
G, MC, MP, L = 9.8, 1.0, 0.1, 0.5
TOTAL_M, PML, TAU = MC + MP, MP * L, 0.02
X_LIM, TH_LIM = 2.4, 0.2094


def _load_strategies():
    from evosax.algorithms import CMA_ES, Sep_CMA_ES
    from evosax.algorithms.distribution_based.open_es import Open_ES
    from evosax.algorithms.population_based.simple_ga import SimpleGA
    return {"simple_ga": SimpleGA, "open_es": Open_ES, "cma_es": CMA_ES, "sep_cma_es": Sep_CMA_ES}


def policy_action(flat, obs):
    W1 = flat[:4 * HIDDEN].reshape(4, HIDDEN)
    W2 = flat[4 * HIDDEN:].reshape(HIDDEN, 1)
    return (jnp.tanh(obs @ W1) @ W2)[0]  # scalar; action = >0


def cartpole_return(flat, init):
    def step(carry, _):
        state, done, total = carry
        x, xdot, th, thdot = state
        force = jnp.where(policy_action(flat, state) > 0, 10.0, -10.0)
        ct, st = jnp.cos(th), jnp.sin(th)
        temp = (force + PML * thdot ** 2 * st) / TOTAL_M
        thacc = (G * st - ct * temp) / (L * (4.0 / 3.0 - MP * ct ** 2 / TOTAL_M))
        xacc = temp - PML * thacc * ct / TOTAL_M
        x, xdot = x + TAU * xdot, xdot + TAU * xacc
        th, thdot = th + TAU * thdot, thdot + TAU * thacc
        nstate = jnp.array([x, xdot, th, thdot])
        fail = (jnp.abs(x) > X_LIM) | (jnp.abs(th) > TH_LIM)
        reward = jnp.where(done, 0.0, 1.0)
        return (nstate, done | fail, total + reward), None
    (_, _, total), _ = jax.lax.scan(step, (init, False, 0.0), None, length=MAX_STEPS)
    return total


def build_fitness_fn(init):
    return jax.jit(jax.vmap(lambda flat: cartpole_return(flat, init)))


def run_evosax(algo, fitness_fn, pop_size, n_gens, seed, log_every=20):
    StrategyCls = _load_strategies()[algo]
    strat = StrategyCls(population_size=pop_size, solution=jnp.zeros(N_PARAMS))
    params = strat.default_params
    rng = jax.random.PRNGKey(seed)
    rng, ri = jax.random.split(rng)
    try:
        state = strat.init(ri, jnp.zeros(N_PARAMS), params)
    except TypeError:
        rng, ri2 = jax.random.split(rng)
        state = strat.init(ri2, jax.random.normal(ri, (pop_size, N_PARAMS)) * 0.5, jnp.zeros(pop_size), params)

    hist_pop = np.zeros((n_gens, pop_size, N_PARAMS), dtype=np.float32)
    hist_fit = np.zeros((n_gens, pop_size), dtype=np.float32)
    t0 = time.time()
    for g in range(n_gens):
        rng, ra, rt = jax.random.split(rng, 3)
        pop, state = strat.ask(ra, state, params)
        reward = np.asarray(fitness_fn(pop))            # steps balanced, maximize
        state, _ = strat.tell(rt, pop, jnp.array(-reward), state, params)  # evosax minimizes
        hist_pop[g] = np.asarray(pop); hist_fit[g] = reward
        if g % log_every == 0 or g == n_gens - 1:
            print(f"  [{algo}] gen {g:3d}  best={reward.max():.0f}  mean={reward.mean():.1f}  ({time.time()-t0:.1f}s)")
    return hist_pop, hist_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="open_es")
    ap.add_argument("--all_algos", action="store_true")
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--gens", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=Path, default=Path("runs"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    init = jnp.array([0.0, 0.0, 0.05, 0.0])  # fixed small pole tilt → deterministic fitness
    fitness_fn = build_fitness_fn(init)
    algos = list(_load_strategies()) if args.all_algos else [args.algo]
    for algo in algos:
        print(f"=== cartpole/{algo} (seed {args.seed}) ===")
        pops, fits = run_evosax(algo, fitness_fn, args.pop, args.gens, args.seed)
        out = args.out_dir / f"cartpole_{algo}_seed{args.seed}.npz"
        np.savez_compressed(out, populations=pops, fitnesses=fits)
        print(f"  saved -> {out}  (best {fits[-1].max():.0f}/200)")


if __name__ == "__main__":
    main()
