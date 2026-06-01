"""Exp 20: temporal-evolution GIFs of the joint UMAP — search *dynamics*.

Where Exp 18c/19 rotate a static 3-D cloud, this animates time: frame g shows each
algorithm's population at generation g (colored, large) over a faint cloud of the
whole run (context). You watch each ES march through the shared weight space —
Simple GA scattering, OpenES staying compact, CMA-ES sweeping along its trajectory.

Reuses the cached 2-D joint embeddings (Exp 18/19) and reconstructs the per-point
generation index from the run files (same stacking order).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.shared import CACHE_DIR, FIGURES_DIR, ensure_dirs, set_science_style

ALGOS = ["simple_ga", "open_es", "cma_es", "sep_cma_es"]
ALGO_LABELS = {"simple_ga": "Simple GA", "open_es": "OpenES",
               "cma_es": "CMA-ES", "sep_cma_es": "sep-CMA-ES"}
ALGO_COLORS = {"simple_ga": "#1f77b4", "open_es": "#ff7f0e",
               "cma_es": "#2ca02c", "sep_cma_es": "#d62728"}
BRAX_RUNS = ROOT.parent / "brax" / "runs"

# benchmark -> (run filename prefix, cached 2-D embedding npz)
BENCHMARKS = {
    "halfcheetah": ("{algo}_seed42.npz", CACHE_DIR / "exp18_multiseed_s42_2d.npz"),
    "make_moons":  ("moons_{algo}_seed42.npz", CACHE_DIR / "exp19_joint_s42_2d.npz"),
}


def reconstruct_labels(run_tpl: str):
    """Rebuild (names, gen_idx) in the exact stacking order used for the fit."""
    names, gen = [], []
    for algo in ALGOS:
        f = BRAX_RUNS / run_tpl.format(algo=algo)
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        G, P, _ = d["populations"].shape
        names += [algo] * (G * P)
        gen.append(np.repeat(np.arange(G), P))
    return np.array(names), np.concatenate(gen)


def make_gif(benchmark: str):
    run_tpl, emb_path = BENCHMARKS[benchmark]
    if not emb_path.exists():
        print(f"  skip {benchmark}: missing {emb_path.name} (run exp18/19 first)")
        return
    emb = np.load(emb_path)["emb"]
    names, gen = reconstruct_labels(run_tpl)
    n_gens = int(gen.max()) + 1

    x0, x1 = emb[:, 0].min(), emb[:, 0].max()
    y0, y1 = emb[:, 1].min(), emb[:, 1].max()
    pad_x, pad_y = 0.05 * (x1 - x0), 0.05 * (y1 - y0)

    fig, ax = plt.subplots(figsize=(7, 6.5))

    def update(g):
        ax.clear()
        ax.set_facecolor("white")
        # faint context: whole run
        ax.scatter(emb[:, 0], emb[:, 1], s=2, alpha=0.06, color="#999999", edgecolors="none")
        # current generation population per algorithm
        cur = gen == g
        for algo in ALGOS:
            m = cur & (names == algo)
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1], s=14, alpha=0.85, edgecolors="none",
                           color=ALGO_COLORS[algo], label=ALGO_LABELS[algo])
        ax.set_xlim(x0 - pad_x, x1 + pad_x)
        ax.set_ylim(y0 - pad_y, y1 + pad_y)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(markerscale=1.4, fontsize=8, loc="upper right")
        ax.set_title(f"{benchmark} — joint UMAP, generation {g:2d}/{n_gens - 1}", fontsize=11)
        return ()

    anim = FuncAnimation(fig, update, frames=n_gens, blit=False)
    out = FIGURES_DIR / f"exp20_{benchmark}_evolution.gif"
    anim.save(out, writer=PillowWriter(fps=12), dpi=110)
    plt.close(fig)
    print(f"Saved {out}  ({n_gens} frames)")


def main():
    ensure_dirs()
    set_science_style()
    for b in BENCHMARKS:
        make_gif(b)


if __name__ == "__main__":
    main()
