"""Roll out saved HalfCheetah populations to extract behavioral descriptors.

Reuses the weights already saved in runs/simple_ga_seed{seed}.npz (no re-evolution):
each individual is rolled out for one episode and we record its x_velocity time
series (gait profile) plus final x_position. These behavioral descriptors are the
basis for a behavior-space UMAP (à la novelty search / MAP-Elites), to compare
against the weight-space UMAP. Output is consumed by the supplementary pixi env.

Usage: .venv/bin/python extract_behavior.py --seed 42
"""
from __future__ import annotations
import argparse, os

import numpy as np
import jax, jax.numpy as jnp
from brax import envs

from neuroevolve_brax import PolicySpec


def make_behavior_rollout(env, spec: PolicySpec, episode_length: int, n_samples: int):
    """vmapped rollout returning (downsampled x_velocity[n_samples], final_x_pos)."""
    stride = max(1, episode_length // n_samples)

    def single(key, params):
        state = env.reset(key)

        def step_fn(carry, _):
            st = carry
            action = spec.forward(params, st.obs)
            nst = env.step(st, action)
            ctrl_cost = jnp.sum(action ** 2)          # energy spent this step
            return nst, (nst.metrics["x_velocity"], nst.metrics["x_position"],
                         ctrl_cost, nst.metrics["reward_run"], nst.metrics["reward_ctrl"])

        _, (xvel, xpos, ctrl, r_run, r_ctrl) = jax.lax.scan(
            step_fn, state, None, length=episode_length)
        xvel_ds = xvel[::stride][:n_samples]          # gait profile
        return xvel_ds, xpos[-1], ctrl.mean(), r_run.sum(), r_ctrl.sum()

    return jax.jit(jax.vmap(single, in_axes=(0, 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episode_len", type=int, default=200)
    ap.add_argument("--n_samples", type=int, default=25)
    ap.add_argument("--algo", default="simple_ga")
    args = ap.parse_args()

    run = f"runs/{args.algo}_seed{args.seed}.npz"
    data = np.load(run, allow_pickle=True)
    pops = data["populations"]            # (gens, pop, n_params)
    fits = data["fitnesses"]              # (gens, pop)
    n_gens, pop_size, n_params = pops.shape

    env = envs.create("halfcheetah", episode_length=args.episode_len, backend="generalized")
    spec = PolicySpec(obs_dim=env.observation_size, hidden_dim=16, act_dim=env.action_size)
    assert spec.n_params == n_params, f"{spec.n_params} != {n_params}"

    roll = make_behavior_rollout(env, spec, args.episode_len, args.n_samples)

    behavior = np.zeros((n_gens, pop_size, args.n_samples), dtype=np.float32)
    final_x = np.zeros((n_gens, pop_size), dtype=np.float32)
    ctrl_cost = np.zeros((n_gens, pop_size), dtype=np.float32)
    reward_run = np.zeros((n_gens, pop_size), dtype=np.float32)
    reward_ctrl = np.zeros((n_gens, pop_size), dtype=np.float32)
    for g in range(n_gens):
        keys = jax.random.split(jax.random.PRNGKey(args.seed * 100000 + g), pop_size)
        xvel, xpos, ctrl, rrun, rctrl = roll(keys, jnp.asarray(pops[g]))
        behavior[g] = np.asarray(xvel)
        final_x[g] = np.asarray(xpos)
        ctrl_cost[g] = np.asarray(ctrl)
        reward_run[g] = np.asarray(rrun)
        reward_ctrl[g] = np.asarray(rctrl)
        if g % 10 == 0 or g == n_gens - 1:
            print(f"  gen {g:3d}  mean|xvel|={np.abs(behavior[g]).mean():6.3f}  "
                  f"final_x={final_x[g].mean():7.3f}  ctrl={ctrl_cost[g].mean():6.3f}")

    out = f"runs/{args.algo}_seed{args.seed}_behavior.npz"
    np.savez_compressed(out, behavior=behavior, final_x=final_x, ctrl_cost=ctrl_cost,
                        reward_run=reward_run, reward_ctrl=reward_ctrl, fitnesses=fits)
    print(f"saved -> {out}  behavior{behavior.shape}")


if __name__ == "__main__":
    main()
