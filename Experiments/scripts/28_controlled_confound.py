"""Exp 28: disentangling the multimodality confound (controlled ablation).

The paper observes that Make Moons (d_w=48, hidden=16, σ=0.11) yields compact
single-basin flow while CIFAR-10 (d_w≈197k, hidden=64, σ=0.05) yields multimodal
structure — but the two differ in input dimension, width AND mutation σ at once.
The paper explicitly leaves disentangling to future work. We do it here: starting
from a fixed base (d_in=2, hidden=16, σ=0.11), we vary ONE factor at a time and
measure the resulting multimodality with the paper's own metrics (aligned-UMAP
embedding spread σ₁ and DBSCAN attractor count), across 3 seeds.

Task is a noise-padded two-moons (only the first 2 input dims are informative), so
input dimensionality can be raised while holding task difficulty roughly fixed.
No PCA anywhere — the swept factor is the only thing that changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


from benchmarks.base import NeuroEvoBase
from utils import compute_aligned_umap_embedding
from shared import (
    CACHE_DIR, FIGURES_DIR, RESULTS_DIR, ensure_dirs, set_science_style,
    count_attractors_dbscan, save_results_csv,
)

POP, N_GENS = 50, 80
SEEDS = [42, 7, 123]
BASE = dict(d_in=2, hidden=16, sigma=0.11)
SWEEPS = {
    "mutation σ":      ("sigma",  [0.03, 0.05, 0.11, 0.20]),
    "hidden width":    ("hidden", [8, 16, 32, 64]),
    "input dimension": ("d_in",   [2, 10, 50, 200]),
}


class ControlledMoons(NeuroEvoBase):
    """Noise-padded two-moons classifier with configurable input dim / width / σ."""

    def __init__(self, d_in, hidden, sigma, pop_size=POP, seed=42):
        self.rng = np.random.default_rng(seed)
        self.pop_size = pop_size
        self.mutation_rate = sigma
        self.weight_init_mean, self.weight_init_std = 0.0, 0.5
        X, y = make_moons(n_samples=1000, noise=0.25, random_state=seed)
        X = StandardScaler().fit_transform(X)
        if d_in > 2:  # pad with standardized noise dims (uninformative)
            noise = self.rng.normal(0, 1, size=(X.shape[0], d_in - 2))
            X = np.hstack([X, noise])
        self.X = X
        self.y = y.reshape(-1, 1).astype(float)
        self.shapes = [(d_in, hidden), (hidden, 1)]
        self.population = [self.random_individual() for _ in range(pop_size)]

    def evaluate(self, ind):
        w1, w2 = ind
        h = np.tanh(self.X @ w1)
        p = 1.0 / (1.0 + np.exp(-(h @ w2)))
        eps = 1e-8
        loss = -(self.y * np.log(p + eps) + (1 - self.y) * np.log(1 - p + eps)).mean()
        return -float(loss)


def run_config(d_in, hidden, sigma, seed):
    key = f"exp28_d{d_in}_h{hidden}_s{sigma}_seed{seed}"
    p = CACHE_DIR / f"{key}.npz"
    if p.exists():
        d = np.load(p)
        return float(d["spread1"]), int(d["attractors"])
    env = ControlledMoons(d_in, hidden, sigma, seed=seed)
    weights_by_gen = []
    for _ in range(N_GENS):
        weights_by_gen.append(np.stack([env.flatten(i) for i in env.population]))
        env.evolve_one_generation()
    emb_all, _, _ = compute_aligned_umap_embedding(weights_by_gen, lambda_align=0.8, random_state=seed)
    spread1 = float(np.std(emb_all[:, 0]))
    attractors = count_attractors_dbscan(emb_all, min_samples=5)
    np.savez_compressed(p, spread1=spread1, attractors=attractors)
    return spread1, attractors


def main():
    ensure_dirs()
    set_science_style()

    rows = []
    results = {}  # (factor_label) -> list of (value, mean_spread, std_spread, mean_attr)
    for label, (factor, values) in SWEEPS.items():
        series = []
        for v in values:
            cfg = dict(BASE)
            cfg[factor] = v
            sp, at = [], []
            for seed in SEEDS:
                s, a = run_config(cfg["d_in"], cfg["hidden"], cfg["sigma"], seed)
                sp.append(s); at.append(a)
                rows.append({"factor": label, "value": v, "seed": seed,
                             "d_in": cfg["d_in"], "hidden": cfg["hidden"], "sigma": cfg["sigma"],
                             "spread1": round(s, 4), "attractors": a})
            series.append((v, float(np.mean(sp)), float(np.std(sp)), float(np.mean(at))))
            print(f"  {label:16s} = {v:<5}  spread σ₁={np.mean(sp):.2f}±{np.std(sp):.2f}  attractors={np.mean(at):.1f}")
        results[label] = series
    save_results_csv(rows, RESULTS_DIR / "exp28_controlled_confound.csv")

    # range of spread induced by each factor = how much it drives multimodality
    print("\nSpread σ₁ range induced by each factor (↑ = stronger driver of multimodality):")
    drivers = {}
    for label, series in results.items():
        means = [m for _, m, _, _ in series]
        drivers[label] = max(means) - min(means)
        print(f"  {label:16s}: Δσ₁ = {drivers[label]:.2f}  ({min(means):.2f} → {max(means):.2f})")
    winner = max(drivers, key=drivers.get)
    print(f"\n→ Dominant driver of multimodality: {winner}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, (label, series) in zip(axes, results.items()):
        vals = [v for v, _, _, _ in series]
        means = np.array([m for _, m, _, _ in series])
        stds = np.array([s for _, _, s, _ in series])
        ax.errorbar(range(len(vals)), means, yerr=stds, marker="o", lw=1.8, capsize=3, color="#1f4e79")
        ax.set_xticks(range(len(vals))); ax.set_xticklabels([str(v) for v in vals])
        ax.set_xlabel(label); ax.set_title(f"{label}\nΔσ₁ = {drivers[label]:.2f}", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("embedding spread σ₁\n(↑ = more multimodal)")
    fig.suptitle("Controlled ablation: which factor drives multimodality? (noise-padded moons, 3 seeds)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "exp28_controlled_confound.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
