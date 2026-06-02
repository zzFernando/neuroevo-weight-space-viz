"""
Neuroevolution on Brax HalfCheetah, saving the entire population
at every generation for downstream UMAP / velocity-field analysis.

Design:
  - Small MLP policy (obs -> hidden -> action) with tanh nonlinearity.
  - Mutation-only Genetic Algorithm with elite truncation, matching the
    setup used in the paper for Make Moons and CIFAR-10.
  - All policy parameters from every individual at every generation are
    written to a single .npz file alongside fitness values, so the UMAP
    analysis is identical to what the paper already does.

Usage:
  python neuroevolve_brax.py --seed 42 --out runs/halfcheetah_seed42.npz
"""

from __future__ import annotations
import argparse, time, os
from dataclasses import dataclass

import numpy as np
import jax, jax.numpy as jnp
from brax import envs

from paths import RUNS_DIR


# ---------------------------------------------------------------------------
# Policy: small MLP, flattened parameter vector
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicySpec:
    obs_dim: int
    hidden_dim: int
    act_dim: int

    @property
    def n_params(self) -> int:
        return (self.obs_dim * self.hidden_dim + self.hidden_dim
                + self.hidden_dim * self.act_dim + self.act_dim)

    def unflatten(self, flat: jnp.ndarray):
        """Split a flat (n_params,) vector into (W1, b1, W2, b2)."""
        o, h, a = self.obs_dim, self.hidden_dim, self.act_dim
        i = 0
        W1 = flat[i:i + o*h].reshape(h, o); i += o*h
        b1 = flat[i:i + h]; i += h
        W2 = flat[i:i + h*a].reshape(a, h); i += h*a
        b2 = flat[i:i + a]
        return W1, b1, W2, b2

    def forward(self, flat, obs):
        W1, b1, W2, b2 = self.unflatten(flat)
        h = jnp.tanh(W1 @ obs + b1)
        return jnp.tanh(W2 @ h + b2)


# ---------------------------------------------------------------------------
# Rollout: scan over env steps, vmap over a population
# ---------------------------------------------------------------------------

def make_rollout_fn(env, spec: PolicySpec, episode_length: int):

    def single_rollout(key, params):
        state = env.reset(key)
        def step_fn(carry, _):
            state = carry
            action = spec.forward(params, state.obs)
            new_state = env.step(state, action)
            return new_state, new_state.reward
        _, rewards = jax.lax.scan(step_fn, state, None, length=episode_length)
        return rewards.sum()

    return jax.jit(jax.vmap(single_rollout, in_axes=(0, 0)))


# ---------------------------------------------------------------------------
# Mutation-only GA matching the paper's setup
# ---------------------------------------------------------------------------

def evolve(rollout_fn, n_params, pop_size, n_gens, sigma, elite_frac,
           seed, episode_length, log_every=10):

    rng = np.random.default_rng(seed)
    # initial population: small random
    pop = rng.normal(0.0, 0.1, size=(pop_size, n_params)).astype(np.float32)

    n_elite = max(2, int(elite_frac * pop_size))
    print(f"[evolve] pop={pop_size}, elites={n_elite}, sigma={sigma}, "
          f"params={n_params}, gens={n_gens}, ep_len={episode_length}")

    history_pop = np.zeros((n_gens, pop_size, n_params), dtype=np.float32)
    history_fit = np.zeros((n_gens, pop_size), dtype=np.float32)

    t_start = time.time()
    for g in range(n_gens):
        # Evaluate
        keys = jax.random.split(jax.random.PRNGKey(seed * 100000 + g), pop_size)
        fitnesses = np.asarray(rollout_fn(keys, jnp.asarray(pop)))

        history_pop[g] = pop
        history_fit[g] = fitnesses

        # Truncation selection + mutation
        order = np.argsort(-fitnesses)        # descending
        elites = pop[order[:n_elite]]
        # children = mutated copies of randomly-picked elites
        parent_idx = rng.integers(0, n_elite, size=pop_size - n_elite)
        children = elites[parent_idx] + rng.normal(
            0.0, sigma, size=(pop_size - n_elite, n_params)
        ).astype(np.float32)
        pop = np.concatenate([elites, children], axis=0)

        if g % log_every == 0 or g == n_gens - 1:
            elapsed = time.time() - t_start
            print(f"  gen {g:4d}  best={fitnesses.max():8.2f}  "
                  f"mean={fitnesses.mean():8.2f}  ({elapsed:.1f}s)")

    return history_pop, history_fit


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env',         default='halfcheetah')
    p.add_argument('--hidden',      type=int,   default=16)
    p.add_argument('--pop',         type=int,   default=50)
    p.add_argument('--gens',        type=int,   default=80)
    p.add_argument('--episode_len', type=int,   default=200)
    p.add_argument('--sigma',       type=float, default=0.05)
    p.add_argument('--elite_frac',  type=float, default=0.2)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--out',         default=str(RUNS_DIR / 'halfcheetah_seed42.npz'))
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    env = envs.create(args.env, episode_length=args.episode_len,
                      backend='generalized')
    spec = PolicySpec(obs_dim=env.observation_size,
                      hidden_dim=args.hidden,
                      act_dim=env.action_size)
    print(f"[main] env={args.env}  obs={spec.obs_dim}  "
          f"hidden={spec.hidden_dim}  act={spec.act_dim}  "
          f"n_params={spec.n_params}")

    rollout_fn = make_rollout_fn(env, spec, args.episode_len)

    # Warm up JIT (one tiny call so timing in evolve() is honest)
    dummy_keys = jax.random.split(jax.random.PRNGKey(0), args.pop)
    dummy_pop = jnp.zeros((args.pop, spec.n_params))
    _ = rollout_fn(dummy_keys, dummy_pop).block_until_ready()

    pops, fits = evolve(
        rollout_fn, spec.n_params,
        pop_size=args.pop, n_gens=args.gens,
        sigma=args.sigma, elite_frac=args.elite_frac,
        seed=args.seed, episode_length=args.episode_len,
    )

    np.savez_compressed(
        args.out,
        populations=pops,        # (gens, pop, n_params)
        fitnesses=fits,          # (gens, pop)
        meta=dict(env=args.env, hidden=args.hidden, pop=args.pop,
                  gens=args.gens, sigma=args.sigma,
                  elite_frac=args.elite_frac, seed=args.seed,
                  n_params=spec.n_params,
                  obs_dim=spec.obs_dim, act_dim=spec.act_dim,
                  episode_len=args.episode_len),
    )
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[main] saved -> {args.out} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
