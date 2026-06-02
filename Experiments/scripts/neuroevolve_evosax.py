"""
neuroevolve_evosax.py — Neuroevolution on Brax HalfCheetah via evosax strategies.

Fixed topology: obs(17) → tanh → hidden(16) → tanh → action(6)   d_w = 390

Supported algorithms (--algo):
  simple_ga    SimpleGA   — mutation + truncation selection
  open_es      Open_ES    — OpenAI ES with antithetic sampling
  cma_es       CMA_ES     — Covariance Matrix Adaptation ES
  sep_cma_es   Sep_CMA_ES — Separable CMA-ES (O(d), good for d_w=390)
  snes         SNES       — Separable Natural ES
  xnes         xNES       — Exponential Natural ES
  pgpe         PGPE       — Parameter-Exploring Policy Gradients

Install:
    pip install "jax[cpu]" brax evosax

Usage:
    python neuroevolve_evosax.py --algo open_es --seed 42
    python neuroevolve_evosax.py --all_algos --seed 42
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from brax import envs

# ── evosax 0.2.x strategy registry ───────────────────────────────────────────

def _load_strategies():
    from evosax.algorithms import CMA_ES, Sep_CMA_ES, PGPE
    from evosax.algorithms.distribution_based.open_es  import Open_ES
    from evosax.algorithms.distribution_based.snes     import SNES
    from evosax.algorithms.distribution_based.xnes     import xNES
    from evosax.algorithms.population_based.simple_ga  import SimpleGA
    return {
        "simple_ga":  SimpleGA,
        "open_es":    Open_ES,
        "cma_es":     CMA_ES,
        "sep_cma_es": Sep_CMA_ES,
        "snes":       SNES,
        "xnes":       xNES,
        "pgpe":       PGPE,
    }


ALGO_DESCRIPTIONS = {
    "simple_ga":  "Simple GA   — mutation + truncation selection",
    "open_es":    "Open ES     — OpenAI ES with antithetic sampling",
    "cma_es":     "CMA-ES      — Covariance Matrix Adaptation ES",
    "sep_cma_es": "Sep-CMA-ES  — Separable CMA-ES (O(d), good for d_w=390)",
    "snes":       "SNES        — Separable Natural ES",
    "xnes":       "xNES        — Exponential Natural ES",
    "pgpe":       "PGPE        — Parameter-Exploring Policy Gradients",
}

# ── MLP policy + rollout ──────────────────────────────────────────────────────

def build_rollout_fn(env, obs_dim: int, hidden_dim: int, act_dim: int,
                     episode_length: int):
    """Returns JIT-vmapped rollout: (keys, pop) → fitness (pop,)."""

    def single_rollout(key, flat):
        h, o, a = hidden_dim, obs_dim, act_dim
        i = 0
        W1 = flat[i : i + o * h].reshape(h, o); i += o * h
        b1 = flat[i : i + h];                   i += h
        W2 = flat[i : i + h * a].reshape(a, h); i += h * a
        b2 = flat[i : i + a]

        state = env.reset(key)

        def step(carry, _):
            s   = carry
            act = jnp.tanh(W2 @ jnp.tanh(W1 @ s.obs + b1) + b2)
            return env.step(s, act), s.reward

        _, rewards = jax.lax.scan(step, state, None, length=episode_length)
        return rewards.sum()

    return jax.jit(jax.vmap(single_rollout, in_axes=(0, 0)))


# ── evosax 0.2.x training loop ────────────────────────────────────────────────

def run_evosax(
    algo_name: str,
    rollout_fn,
    n_params: int,
    pop_size: int,
    n_gens: int,
    seed: int,
    log_every: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (populations G×P×D, fitnesses G×P)."""
    STRATEGIES = _load_strategies()
    StrategyCls = STRATEGIES[algo_name]

    solution_template = jnp.zeros(n_params)
    strategy = StrategyCls(population_size=pop_size, solution=solution_template)
    params   = strategy.default_params

    rng = jax.random.PRNGKey(seed)

    # init — SimpleGA (population-based) needs (key, population, fitness, params)
    # distribution-based needs (key, mean, params)
    rng, rng_init = jax.random.split(rng)
    try:
        # distribution-based
        state = strategy.init(rng_init, jnp.zeros(n_params), params)
    except TypeError:
        # population-based: needs initial population + fitness
        init_pop = jax.random.normal(rng_init, (pop_size, n_params)) * 0.1
        init_fit = jnp.zeros(pop_size)
        rng, rng_init2 = jax.random.split(rng)
        state = strategy.init(rng_init2, init_pop, init_fit, params)

    # JIT warmup
    rng, rng_w = jax.random.split(rng)
    pop_w, _ = strategy.ask(rng_w, state, params)
    keys_w   = jax.random.split(rng_w, pop_size)
    _ = rollout_fn(keys_w, pop_w).block_until_ready()
    print(f"  JIT warmup done. pop_size={pop_size}  n_params={n_params}")

    history_pop = np.zeros((n_gens, pop_size, n_params), dtype=np.float32)
    history_fit = np.zeros((n_gens, pop_size),           dtype=np.float32)
    t0 = time.time()

    for g in range(n_gens):
        rng, rng_ask, rng_eval, rng_tell = jax.random.split(rng, 4)

        pop, state   = strategy.ask(rng_ask, state, params)
        keys         = jax.random.split(rng_eval, pop_size)
        fitness      = np.asarray(rollout_fn(keys, pop))    # (pop,) reward, higher = better
        # evosax MINIMIZES the value passed to tell; we want to MAXIMIZE reward,
        # so feed it -fitness. (history keeps the true reward, higher = better.)
        state, _     = strategy.tell(rng_tell, pop, jnp.array(-fitness), state, params)

        history_pop[g] = np.asarray(pop)
        history_fit[g] = fitness

        if g % log_every == 0 or g == n_gens - 1:
            print(f"  gen {g:4d}/{n_gens}  best={fitness.max():8.2f}"
                  f"  mean={fitness.mean():8.2f}  ({time.time()-t0:.1f}s)")

    return history_pop, history_fit


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Neuroevolution on HalfCheetah via evosax 0.2.x strategies"
    )
    p.add_argument("--algo",        default="open_es",
                   choices=list(ALGO_DESCRIPTIONS))
    p.add_argument("--all_algos",   action="store_true",
                   help="Run all strategies sequentially for this seed")
    p.add_argument("--env",         default="halfcheetah")
    p.add_argument("--hidden",      type=int,  default=16)
    p.add_argument("--pop",         type=int,  default=50)
    p.add_argument("--gens",        type=int,  default=80)
    p.add_argument("--episode_len", type=int,  default=300)
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--out_dir",     type=Path, default=Path("runs"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    env     = envs.create(args.env, episode_length=args.episode_len,
                          backend="generalized")
    obs_dim = env.observation_size
    act_dim = env.action_size
    n_params = obs_dim * args.hidden + args.hidden + args.hidden * act_dim + act_dim

    print(f"env={args.env}  obs={obs_dim}  act={act_dim}"
          f"  hidden={args.hidden}  d_w={n_params}")
    print(f"pop={args.pop}  gens={args.gens}  ep_len={args.episode_len}")

    rollout_fn = build_rollout_fn(env, obs_dim, args.hidden, act_dim,
                                  args.episode_len)

    algos = list(ALGO_DESCRIPTIONS) if args.all_algos else [args.algo]

    for algo in algos:
        out = args.out_dir / f"{algo}_seed{args.seed}.npz"
        print(f"\n{'='*60}")
        print(f"  {ALGO_DESCRIPTIONS[algo]}")
        print(f"  seed={args.seed}  →  {out}")
        print(f"{'='*60}")

        pops, fits = run_evosax(
            algo_name=algo,
            rollout_fn=rollout_fn,
            n_params=n_params,
            pop_size=args.pop,
            n_gens=args.gens,
            seed=args.seed,
        )

        np.savez_compressed(
            out,
            populations=pops,   # (gens, pop, n_params)
            fitnesses=fits,      # (gens, pop)
        )
        size_mb = out.stat().st_size / 1e6
        print(f"  saved → {out}  ({size_mb:.1f} MB)")
        print(f"  best fitness (final gen): {fits[-1].max():.2f}")


if __name__ == "__main__":
    main()
