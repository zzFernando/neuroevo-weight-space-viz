"""Exp 16: 3-D UMAP of HalfCheetah — does the extra dimension help organize fitness?

Exp 15 found the rich full-behavior descriptor (gait+dist+effort) organizes fitness
best in 2-D (k-NN corr 0.35 vs 0.26 weight-space). A 3rd embedding dimension can
disentangle structure that overlaps in 2-D. Here we build 3-D UMAPs of both the
weight space and the full-behavior space, color by fitness, and re-measure k-NN
fitness coherence to check whether 3-D actually helps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors


from utils import compute_aligned_umap_embedding
from shared import CACHE_DIR, FIGURES_DIR, ensure_dirs, set_science_style

SEED = 42
LAMBDA_ALIGN = 0.8
from paths import RUNS_DIR as BRAX_RUNS


def zscore_by_gen(arr_by_gen):
    stacked = np.vstack(arr_by_gen)
    mu, sd = stacked.mean(0), stacked.std(0) + 1e-9
    return [(a - mu) / sd for a in arr_by_gen]


def embed(name, bd_by_gen, n_components):
    p = CACHE_DIR / f"exp16_emb_{name}_{n_components}d_{SEED}.npz"
    if p.exists():
        return np.load(p, allow_pickle=True)["emb_all"]
    emb_all, _, _ = compute_aligned_umap_embedding(
        bd_by_gen, lambda_align=LAMBDA_ALIGN, random_state=SEED,
        metric="euclidean", n_components=n_components)
    np.savez_compressed(p, emb_all=emb_all)
    return emb_all


def knn_corr(emb, fit, k=15):
    f = (fit - fit.min()) / (fit.max() - fit.min() + 1e-9)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    _, idx = nn.kneighbors(emb)
    idx = idx[:, 1:]
    dev = float(np.abs(f[:, None] - f[idx]).mean())
    corr = float(np.corrcoef(f, f[idx].mean(1))[0, 1])
    return dev, corr


def load_weight_by_gen():
    pops = np.load(BRAX_RUNS / f"simple_ga_seed{SEED}.npz", allow_pickle=True)["populations"]
    return [pops[g].astype(np.float64) for g in range(pops.shape[0])]


def main():
    ensure_dirs()
    set_science_style()
    d = np.load(BRAX_RUNS / f"simple_ga_seed{SEED}_behavior.npz", allow_pickle=True)
    G = d["behavior"].shape[0]
    full = np.concatenate([d["behavior"], d["final_x"][..., None], d["ctrl_cost"][..., None]], -1)
    full_by_gen = zscore_by_gen([full[g].astype(np.float64) for g in range(G)])
    weight_by_gen = load_weight_by_gen()
    fit = d["fitnesses"].ravel().astype(float)

    spaces = {
        "weight-space": weight_by_gen,
        "full-behavior": full_by_gen,
    }

    # compute 2-D and 3-D, report coherence
    print("k-NN fitness coherence (corr ↑ / dev ↓):")
    emb3 = {}
    for name, bd in spaces.items():
        for nc in (2, 3):
            e = embed(name, bd, nc)
            f = fit[:len(e)]
            dev, corr = knn_corr(e, f)
            print(f"  {name:14s} {nc}D   corr={corr:.3f}  dev={dev:.4f}")
            if nc == 3:
                emb3[name] = (e, f)

    # 3-D figure
    fig = plt.figure(figsize=(15, 6.5), dpi=130)
    fpct = {}
    for i, (name, (emb, f)) in enumerate(emb3.items(), start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        fmin, fmax = np.percentile(f, 50), np.percentile(f, 99)
        c = np.clip((f - fmin) / (fmax - fmin + 1e-9) * 100, 0, 100)
        sc = ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=c, cmap="viridis",
                        s=7, alpha=0.6, edgecolors="none")
        best = int(np.argmax(f))
        ax.scatter(emb[best, 0], emb[best, 1], emb[best, 2], marker="*", s=180,
                   c="#D62728", edgecolors="white", linewidths=0.8)
        _, corr = knn_corr(emb, f)
        ax.set_title(f"{name} — 3D UMAP\nk-NN fitness corr={corr:.2f}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        fig.colorbar(sc, ax=ax, label="fitness pct", shrink=0.55, pad=0.04)

    fig.suptitle(f"HalfCheetah 3-D UMAP colored by fitness (seed={SEED}, ★ = best)",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    out = FIGURES_DIR / "exp16_3d_embedding.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
