"""Exp 14: behavior-space vs weight-space UMAP for HalfCheetah, colored by fitness.

Exp 13 showed weight-space UMAP fails to organize HalfCheetah individuals by
fitness — because the weight→fitness map of an RL controller is rugged. The
quality-diversity fix is to embed *behavior* instead of weights. Behavioral
descriptors (x_velocity gait profile + final x position) were rolled out by
brax/extract_behavior.py. Here we UMAP those and compare, side by side, against
the weight-space embedding, both colored by fitness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import compute_aligned_umap_embedding
from experiments.shared import (
    set_science_style,
    CACHE_DIR, FIGURES_DIR, ensure_dirs,
    count_attractors_dbscan, plot_fitness_panel, temporal_coherence,
)

SEED = 42
LAMBDA_ALIGN = 0.8
BRAX_RUNS = ROOT.parent / "brax" / "runs"


def load_behavior():
    p = BRAX_RUNS / f"simple_ga_seed{SEED}_behavior.npz"
    d = np.load(p, allow_pickle=True)
    beh = d["behavior"]                          # (gens, pop, n_samples)
    final_x = d["final_x"]                        # (gens, pop)
    fits = d["fitnesses"]                         # (gens, pop)
    # behavior descriptor = gait profile + distance traveled
    bd = np.concatenate([beh, final_x[..., None]], axis=-1).astype(np.float64)
    return [bd[g] for g in range(bd.shape[0])], fits.ravel().astype(float)


def load_weight_embedding():
    """Weight-space euclidean embedding cached by Exp 12."""
    p = CACHE_DIR / f"exp12_emb_halfcheetah_{SEED}_euclidean.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return {"emb_all": d["emb_all"], "per_gen": list(d["per_gen"])}


def behavior_embedding(bd_by_gen):
    p = CACHE_DIR / f"exp14_behavior_emb_{SEED}.npz"
    if p.exists():
        d = np.load(p, allow_pickle=True)
        return {"emb_all": d["emb_all"], "per_gen": list(d["per_gen"])}
    emb_all, _, per_gen = compute_aligned_umap_embedding(
        bd_by_gen, lambda_align=LAMBDA_ALIGN, random_state=SEED, metric="euclidean")
    np.savez_compressed(p, emb_all=emb_all, per_gen=np.array(per_gen, dtype=object))
    return {"emb_all": emb_all, "per_gen": list(per_gen)}


def main():
    ensure_dirs()
    set_science_style()
    bd_by_gen, fitness = load_behavior()
    beh = behavior_embedding(bd_by_gen)
    wgt = load_weight_embedding()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=130)
    last_sc = None

    panels = [("Weight-space UMAP\n(390-dim weights, euclidean)", wgt),
              ("Behavior-space UMAP\n(gait profile + distance, euclidean)", beh)]
    for ax, (title, payload) in zip(axes, panels):
        ax.set_facecolor("white")
        if payload is None:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", color="#a00")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=11)
            continue
        emb = payload["emb_all"]
        f = fitness
        if len(f) != len(emb):
            n = min(len(f), len(emb)); emb, f = emb[:n], f[:n]
        last_sc = plot_fitness_panel(ax, emb, f, title=None)
        tc = temporal_coherence(payload["per_gen"])
        at = count_attractors_dbscan(emb, min_samples=5)
        ax.set_title(f"{title}\ntc={tc:.2f}  att={at}", fontsize=11)

    if last_sc is not None:
        fig.colorbar(last_sc, ax=axes, label="fitness percentile", shrink=0.7, pad=0.02)

    fig.suptitle(
        f"HalfCheetah: does behavior-space organize fitness better than weight-space?  "
        f"(seed={SEED}, ★ = best)", fontsize=12, y=1.02)
    out = FIGURES_DIR / "exp14_behavior_vs_weight_space.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
