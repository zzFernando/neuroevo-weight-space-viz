"""
NeuroEvoBrax — HalfCheetah benchmark via Brax/JAX.

Plugs into the paper's NeuroEvoBase framework so that
compute_aligned_umap_embedding, temporal_coherence, and all
Exp 1 metrics work identically to Make Moons and CIFAR-10.

Architecture: obs → tanh → hidden(16) → tanh → action
  obs_dim=17, act_dim=6, hidden=16  →  d_w = 390
  (17×16 + 16 + 6×16 + 6 = 390)

GA: same mutation-only elite selection as base class (sigma=0.05,
elite_frac=0.2), matching paper Table 1 recommended config.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import NeuroEvoBase


def _check_jax():
    try:
        import jax  # noqa: F401
        import brax  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "NeuroEvoBrax requires JAX and Brax.\n"
            "Install with: pip install 'jax[cpu]' brax==0.14.2\n"
            f"Original error: {e}"
        ) from e


class NeuroEvoBrax(NeuroEvoBase):
    """HalfCheetah neuroevolution using Brax physics simulation."""

    # No PCA pre-reduction: d_w=390 goes straight to UMAP (no projection-of-projection)
    pca_dims: int | None = None

    def __init__(
        self,
        pop_size: int = 50,
        hidden_dim: int = 16,
        mutation_rate: float = 0.05,
        seed: int = 42,
        episode_length: int = 300,
        env_name: str = "halfcheetah",
    ) -> None:
        _check_jax()
        import jax
        import jax.numpy as jnp
        from brax import envs

        self.rng = np.random.default_rng(seed)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.weight_init_mean = 0.0
        self.weight_init_std = 0.1
        self._hidden_dim = hidden_dim
        self._episode_length = episode_length
        self._eval_counter = 0
        self._base_key = jax.random.PRNGKey(seed)

        env = envs.create(env_name, episode_length=episode_length, backend="generalized")
        self._env = env
        obs_dim  = env.observation_size
        act_dim  = env.action_size

        # Weight layout: W1(h,o), b1(h,), W2(a,h), b2(a,)
        self.shapes = [
            (hidden_dim, obs_dim),
            (hidden_dim,),
            (act_dim, hidden_dim),
            (act_dim,),
        ]
        self.population = [self.random_individual() for _ in range(pop_size)]

        # JIT-compiled single-individual rollout
        def _single_rollout(key: jax.Array, flat: jax.Array) -> jax.Array:
            o, h, a = obs_dim, hidden_dim, act_dim
            i = 0
            W1 = flat[i : i + o * h].reshape(h, o); i += o * h
            b1 = flat[i : i + h];                   i += h
            W2 = flat[i : i + h * a].reshape(a, h); i += h * a
            b2 = flat[i : i + a]

            state = env.reset(key)

            def step(carry, _):
                s = carry
                obs = s.obs
                act = jnp.tanh(W2 @ jnp.tanh(W1 @ obs + b1) + b2)
                return env.step(s, act), s.reward

            _, rewards = jax.lax.scan(step, state, None, length=episode_length)
            return rewards.sum()

        self._single_rollout = jax.jit(_single_rollout)

        # JIT-compiled vmapped population rollout (used in evolve_one_generation)
        self._batch_rollout = jax.jit(
            jax.vmap(_single_rollout, in_axes=(0, 0))
        )

        # Warm up JIT (avoids timing the first generation)
        import jax
        dummy_keys = jax.random.split(self._base_key, pop_size)
        dummy_pop  = jnp.zeros((pop_size, sum(
            int(np.prod(s)) for s in self.shapes
        )))
        _ = self._batch_rollout(dummy_keys, dummy_pop).block_until_ready()
        print(f"[NeuroEvoBrax] env={env_name}  obs={obs_dim}  act={act_dim}"
              f"  d_w={dummy_pop.shape[1]}  JIT warmup done.")

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, individual: Sequence[np.ndarray]) -> float:
        """Single-individual rollout (JIT-compiled, called sequentially by framework)."""
        import jax
        import jax.numpy as jnp
        key = jax.random.fold_in(self._base_key, self._eval_counter)
        self._eval_counter += 1
        flat = jnp.asarray(self.flatten(individual), dtype=jnp.float32)
        return float(self._single_rollout(key, flat))

    # ── Override evolve_one_generation to use vmapped batch eval ──────────────

    def evolve_one_generation(
        self, elite_frac: float = 0.2, min_elite: int = 2
    ) -> tuple[int, float]:
        """Batch-evaluates the population with vmap instead of sequential evaluate()."""
        import jax
        import jax.numpy as jnp

        pop_flat = np.stack([self.flatten(ind) for ind in self.population]).astype(np.float32)
        keys     = jax.random.split(
            jax.random.fold_in(self._base_key, self._eval_counter), self.pop_size
        )
        self._eval_counter += self.pop_size

        fitness = np.asarray(self._batch_rollout(keys, jnp.asarray(pop_flat)))

        n_elite   = max(min_elite, int(self.pop_size * elite_frac))
        elite_idx = np.argsort(fitness)[-n_elite:]
        elites    = [self.population[i] for i in elite_idx]

        new_pop: list = elites.copy()
        while len(new_pop) < self.pop_size:
            parent = elites[self.rng.integers(0, len(elites))]
            new_pop.append(self.mutate(parent))
        self.population = new_pop

        # Evaluate post-mutation population (needed by run_evolution_benchmark)
        # Store fitness so that subsequent evaluate() calls return cached values
        new_flat = np.stack([self.flatten(ind) for ind in self.population]).astype(np.float32)
        keys2    = jax.random.split(
            jax.random.fold_in(self._base_key, self._eval_counter), self.pop_size
        )
        self._eval_counter += self.pop_size
        new_fitness = np.asarray(self._batch_rollout(keys2, jnp.asarray(new_flat)))

        # Cache so evaluate() doesn't re-run for these same individuals
        self._fitness_cache = {i: float(new_fitness[i]) for i in range(self.pop_size)}

        best_idx = int(np.argmax(new_fitness))
        return best_idx, float(new_fitness[best_idx])

    def evaluate(self, individual: Sequence[np.ndarray]) -> float:  # noqa: F811
        """Returns cached fitness when available (set by evolve_one_generation)."""
        import jax
        import jax.numpy as jnp

        # Try to match individual to cached result by position in current population
        if hasattr(self, "_fitness_cache"):
            for i, pop_ind in enumerate(self.population):
                if all(
                    np.array_equal(a, b)
                    for a, b in zip(individual, pop_ind)
                ):
                    return self._fitness_cache[i]

        # Fallback: run single rollout
        key  = jax.random.fold_in(self._base_key, self._eval_counter)
        self._eval_counter += 1
        flat = jnp.asarray(self.flatten(individual), dtype=jnp.float32)
        return float(self._single_rollout(key, flat))
