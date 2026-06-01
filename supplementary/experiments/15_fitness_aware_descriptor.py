"""Exp 15: fitness-aware behavioral descriptor for HalfCheetah.

Exp 14 showed a naive kinematic descriptor (gait + distance) is anti-correlated
with fitness, because fitness = reward_run + reward_ctrl is dominated by control
cost. Here we build a fitness-aware descriptor that separates the two reward
drivers — net displacement (≈ run) and control effort (≈ ctrl) — and test whether
its UMAP organizes fitness better than weight-space or the naive descriptor.

Compares, via k-NN local fitness coherence, four embeddings:
  weight-space · naive-behavior (gait+dist) · fitness-aware (dist+effort) · full.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import compute_aligned_umap_embedding
from experiments.shared import CACHE_DIR, FIGURES_DIR, ensure_dirs, plot_fitness_panel, set_science_style

SEED = 42
LAMBDA_ALIGN = 0.8
BRAX_RUNS = ROOT.parent / "brax" / "runs"


def zscore_by_gen(arr_by_gen: list[np.ndarray]) -> list[np.ndarray]:
    """Standardize each feature using global stats (keeps gens comparable)."""
    stacked = np.vstack(arr_by_gen)
    mu, sd = stacked.mean(0), stacked.std(0) + 1e-9
    return [(a - mu) / sd for a in arr_by_gen]


def embed(name: str, bd_by_gen: list[np.ndarray]):
    p = CACHE_DIR / f"exp15_emb_{name}_{SEED}.npz"
    if p.exists():
        d = np.load(p, allow_pickle=True)
        return d["emb_all"]
    emb_all, _, _ = compute_aligned_umap_embedding(
        bd_by_gen, lambda_align=LAMBDA_ALIGN, random_state=SEED, metric="euclidean")
    np.savez_compressed(p, emb_all=emb_all)
    return emb_all


def knn_fitness_coherence(emb: np.ndarray, fit: np.ndarray, k: int = 15):
    f = (fit - fit.min()) / (fit.max() - fit.min() + 1e-9)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    _, idx = nn.kneighbors(emb)
    idx = idx[:, 1:]
    local_dev = float(np.abs(f[:, None] - f[idx]).mean())
    corr = float(np.corrcoef(f, f[idx].mean(1))[0, 1])
    return local_dev, corr


def main():
    ensure_dirs()
    set_science_style()
    d = np.load(BRAX_RUNS / f"simple_ga_seed{SEED}_behavior.npz", allow_pickle=True)
    G, P, S = d["behavior"].shape
    gait = d["behavior"]                     # (G,P,25)
    final_x = d["final_x"][..., None]        # (G,P,1)
    ctrl = d["ctrl_cost"][..., None]         # (G,P,1)
    fit = d["fitnesses"].ravel().astype(float)

    def by_gen(x):
        return [x[g].astype(np.float64) for g in range(G)]

    # descriptor variants
    naive = zscore_by_gen(by_gen(np.concatenate([gait, final_x], -1)))
    fitness_aware = zscore_by_gen(by_gen(np.concatenate([final_x, ctrl], -1)))
    full = zscore_by_gen(by_gen(np.concatenate([gait, final_x, ctrl], -1)))

    embeds = {}
    # weight-space from Exp 12
    wp = CACHE_DIR / f"exp12_emb_halfcheetah_{SEED}_euclidean.npz"
    if wp.exists():
        embeds["weight-space"] = np.load(wp, allow_pickle=True)["emb_all"]
    embeds["naive-behavior\n(gait+dist)"] = embed("naive", naive)
    embeds["fitness-aware\n(dist+effort)"] = embed("fitaware", fitness_aware)
    embeds["full-behavior\n(gait+dist+effort)"] = embed("full", full)

    print("k-NN local fitness coherence (lower dev / higher corr = better organized):")
    rows = {}
    for name, emb in embeds.items():
        f = fit
        if len(f) != len(emb):
            n = min(len(f), len(emb)); emb, f = emb[:n], f[:n]
        dev, corr = knn_fitness_coherence(emb, f)
        rows[name] = (dev, corr, emb, f)
        print(f"  {name.replace(chr(10),' '):34s}  dev={dev:.4f}  corr={corr:.3f}")

    # figure
    n = len(embeds)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.6), dpi=130)
    axes = np.atleast_1d(axes)
    last_sc = None
    for ax, (name, (dev, corr, emb, f)) in zip(axes, rows.items()):
        last_sc = plot_fitness_panel(ax, emb, f, title=None)
        ax.set_title(f"{name}\ncorr={corr:.2f}  dev={dev:.3f}", fontsize=10)
    if last_sc is not None:
        fig.colorbar(last_sc, ax=axes, label="fitness percentile", shrink=0.7, pad=0.01)
    fig.suptitle(
        f"HalfCheetah: fitness organization across embedding spaces (seed={SEED}, ★ = best)",
        fontsize=12, y=1.03)
    out = FIGURES_DIR / "exp15_fitness_aware_descriptor.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
