"""
Generate synthetic brax-format .npz files for stress-testing visualizations.

Simulates a realistic GA on a multi-modal landscape in high-dimensional weight
space, producing populations and fitnesses in the exact format that
analyze_umap.py and figure_summary.py expect.

Landscape design
----------------
Curse of dimensionality makes Gaussian fitness collapse in 390D (distance
between random points ≈ σ√D ≈ 158, far larger than any basin radius).

Fix: **low-rank fitness** — only a r-dimensional subspace matters.
  fitness(w) = max_k [ peak_k · exp(-‖Pₖ w - tₖ‖² / 2R²) ]
where Pₖ is a (r=8, D) random orthogonal projection.
In the projected space distances are O(σ√r) ≈ σ·2.8, so gaussians are visible.

Stress parameters
-----------------
  n_gens   = 300   (vs 80 in the paper)
  pop_size = 500   (vs 50 in the paper)
  n_params = 390   (HalfCheetah 16-unit MLP dimensionality)
  n_seeds  = 5
  => 150,000 UMAP points per seed, 750,000 total
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

N_GENS   = 300
POP_SIZE = 500
N_PARAMS = 390
N_SEEDS  = 5
SEEDS    = [42, 123, 7, 31, 99]

N_ATTRACTORS = 6
RANK         = 8     # effective dimensionality of fitness landscape
BASIN_RADIUS = 1.5   # in projected r-dim space
SIGMA_INIT   = 3.0   # initial population spread (in projected space ≈ 3·√8 ≈ 8.5)
SIGMA_MUT    = 0.12  # mutation std — annealed over training
ELITE_FRAC   = 0.15

# Attractor fitness peaks: global opt (1.0) + 5 local optima
ATTRACTOR_FITNESS = np.array([1.0, 0.82, 0.75, 0.68, 0.60, 0.54], dtype=np.float32)

from paths import RUNS_DIR


# ── Landscape ─────────────────────────────────────────────────────────────────

def build_landscape(seed: int):
    """
    Returns projections P (K, r, D) and targets t (K, r).
    fitness(w) = max_k [ peak_k · gaussian(Pₖ @ w, tₖ, R) ]
    """
    rng = np.random.default_rng(seed)

    # Random orthogonal projections via QR (shared subspace structure)
    Q, _ = np.linalg.qr(rng.standard_normal((N_PARAMS, N_PARAMS)))
    Q = Q[:, :RANK * N_ATTRACTORS].T   # (K*r, D)

    projections = Q.reshape(N_ATTRACTORS, RANK, N_PARAMS).astype(np.float32)

    # Targets: spread in projected space at distance ~3–6 from origin
    targets = rng.standard_normal((N_ATTRACTORS, RANK)).astype(np.float32)
    norms = np.linalg.norm(targets, axis=1, keepdims=True)
    scales = rng.uniform(3.0, 6.0, (N_ATTRACTORS, 1)).astype(np.float32)
    targets = targets / norms * scales

    return projections, targets


def fitness_fn(
    pop: np.ndarray,          # (P, D)
    projections: np.ndarray,  # (K, r, D)
    targets: np.ndarray,      # (K, r)
    peaks: np.ndarray,        # (K,)
    noise_std: float = 0.03,
    rng=None,
) -> np.ndarray:
    P = pop.shape[0]
    K = projections.shape[0]

    all_f = np.empty((P, K), dtype=np.float32)
    for k in range(K):
        proj = pop @ projections[k].T   # (P, r)
        diff = proj - targets[k]        # (P, r)
        dist_sq = (diff * diff).sum(axis=1)
        all_f[:, k] = peaks[k] * np.exp(-dist_sq / (2 * BASIN_RADIUS ** 2))

    f = all_f.max(axis=1)
    if rng is not None and noise_std > 0:
        f += rng.standard_normal(P).astype(np.float32) * noise_std
    return f


# ── GA ────────────────────────────────────────────────────────────────────────

def run_ga(
    seed: int,
    projections: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_elite = max(1, int(POP_SIZE * ELITE_FRAC))

    # Initialise: each individual placed near a random attractor in weight space
    # (pseudo-inverse of projection maps target back into weight space)
    k_assign = rng.integers(0, N_ATTRACTORS, POP_SIZE)
    pop = np.empty((POP_SIZE, N_PARAMS), dtype=np.float32)
    for i in range(POP_SIZE):
        k = k_assign[i]
        # Pseudo-inverse: P† t gives a weight vector whose projection = t
        pinv = np.linalg.pinv(projections[k])       # (D, r)
        base = (pinv @ targets[k]).astype(np.float32)
        pop[i] = base + rng.standard_normal(N_PARAMS).astype(np.float32) * SIGMA_INIT

    all_pops = np.empty((N_GENS, POP_SIZE, N_PARAMS), dtype=np.float32)
    all_fits = np.empty((N_GENS, POP_SIZE), dtype=np.float32)

    sigma_schedule = np.linspace(SIGMA_MUT * 2.5, SIGMA_MUT * 0.4, N_GENS).astype(np.float32)

    for g in range(N_GENS):
        fits = fitness_fn(pop, projections, targets, ATTRACTOR_FITNESS, rng=rng)
        all_pops[g] = pop
        all_fits[g] = fits

        elite_idx = np.argsort(fits)[-n_elite:]
        elites = pop[elite_idx]

        parent_idx = rng.integers(0, n_elite, size=POP_SIZE - n_elite)
        children = elites[parent_idx] + (
            rng.standard_normal((POP_SIZE - n_elite, N_PARAMS)).astype(np.float32)
            * sigma_schedule[g]
        )
        pop = np.vstack([elites, children])

        if (g + 1) % 50 == 0:
            print(f"    gen {g+1:3d}/{N_GENS}  "
                  f"mean={fits.mean():.4f}  best={fits.max():.4f}  "
                  f"σ={sigma_schedule[g]:.4f}")

    return all_pops, all_fits


# ── Main ──────────────────────────────────────────────────────────────────────

def main(force: bool = False) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print("Building landscape (shared across seeds)…")
    projections, targets = build_landscape(seed=0)

    total_pts = N_GENS * POP_SIZE * N_SEEDS
    print(f"Stress params: {N_GENS} gens × {POP_SIZE} pop = "
          f"{N_GENS*POP_SIZE:,} points/seed × {N_SEEDS} seeds = {total_pts:,} total\n")

    for seed in SEEDS:
        out = RUNS_DIR / f"stress_seed{seed}.npz"
        if out.exists() and not force:
            print(f"[seed {seed}] already exists — skipping (--force to overwrite)")
            continue

        print(f"[seed {seed}] running GA…")
        t0 = time.time()
        pops, fits = run_ga(seed, projections, targets)
        elapsed = time.time() - t0

        meta = dict(
            env="stress-test",
            seed=seed,
            n_gens=N_GENS,
            pop_size=POP_SIZE,
            n_params=N_PARAMS,
            n_attractors=N_ATTRACTORS,
            rank=RANK,
            elite_frac=ELITE_FRAC,
        )
        np.savez_compressed(out, populations=pops, fitnesses=fits, meta=meta)
        size_mb = out.stat().st_size / 1e6
        print(f"[seed {seed}] done in {elapsed:.1f}s  →  {out.name}  ({size_mb:.0f} MB)\n")

    print("All seeds done.")
    print("\nAnalyse single seed:")
    print("  cd brax && pixi run python analyze_umap.py runs/stress_seed42.npz --out figures/stress_seed42.png")
    print("\nComposite (all seeds):")
    print("  cd brax && pixi run python figure_summary.py --runs_glob 'runs/stress_seed*.npz' --out figures/stress_summary.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    main(force=args.force)
