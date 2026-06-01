"""Exp 12: UMAP distance-metric comparison on the benchmarks where evolution worked.

make_moons (low-dim) and halfcheetah (390-dim brax weights) are the two benchmarks
where neuroevolution actually learned, so they are the trustworthy testbeds for
judging distance metrics. CIFAR-10 is excluded (it barely beat random). Renders the
UMAP projection (scatter by generation) for each candidate metric, side by side.

halfcheetah weights are read from brax/runs/simple_ga_seed{seed}.npz (key
'populations', shape [n_gen, pop, dim]) — no JAX/Brax needed.
"""
from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import compute_aligned_umap_embedding
from experiments.shared import (
    set_science_style,
    CACHE_DIR,
    FIGURES_DIR,
    ensure_dirs,
    count_attractors_dbscan,
    temporal_coherence,
)

logging.basicConfig(filename="errors.log", filemode="a",
                    format="%(asctime)s %(levelname)s %(message)s", level=logging.ERROR)

METRICS = ["euclidean", "mahalanobis", "chebyshev", "cosine", "correlation"]
SEED = 42
LAMBDA_ALIGN = 0.8
DISPLAY = {"make_moons": "Make Moons (low-dim)", "halfcheetah": "HalfCheetah (390-dim)"}
BRAX_RUNS = ROOT.parent / "brax" / "runs"


def load_weights(benchmark: str, seed: int) -> list[np.ndarray] | None:
    """Return weights_by_gen: list of (pop, dim) arrays, one per generation."""
    if benchmark == "make_moons":
        for p in (CACHE_DIR / f"exp9_evo_make_moons_{seed}.npz",
                  CACHE_DIR / f"exp10_evo_make_moons_{seed}.npz"):
            if p.exists():
                d = np.load(p, allow_pickle=True)
                return [np.asarray(w, dtype=np.float64) for w in d["weights_by_gen"]]
        return None
    if benchmark == "halfcheetah":
        p = BRAX_RUNS / f"simple_ga_seed{seed}.npz"
        if not p.exists():
            return None
        pops = np.load(p, allow_pickle=True)["populations"]  # (n_gen, pop, dim)
        return [pops[g].astype(np.float64) for g in range(pops.shape[0])]
    return None


def metric_kwds_for(metric: str, weights_by_gen: list[np.ndarray]) -> dict | None:
    if metric == "mahalanobis":
        feats = np.vstack(weights_by_gen)
        vi = np.linalg.pinv(np.atleast_2d(np.cov(feats, rowvar=False)))
        return {"VI": vi.astype(np.float64)}
    return None


def get_embedding(benchmark: str, metric: str, wbg: list[np.ndarray]) -> dict | None:
    path = CACHE_DIR / f"exp12_emb_{benchmark}_{SEED}_{metric}.npz"
    if path.exists():
        try:
            d = np.load(path, allow_pickle=True)
            return {"emb_all": d["emb_all"], "gen_labels": d["gen_labels"], "per_gen": list(d["per_gen"])}
        except Exception:
            pass
    try:
        kwds = metric_kwds_for(metric, wbg)
        emb_all, gen_labels, per_gen = compute_aligned_umap_embedding(
            wbg, lambda_align=LAMBDA_ALIGN, random_state=SEED, metric=metric, metric_kwds=kwds)
    except Exception as exc:
        logging.error("Exp12 embed failed %s/%s: %s\n%s", benchmark, metric, exc, traceback.format_exc())
        return None
    np.savez_compressed(path, emb_all=emb_all, gen_labels=gen_labels,
                        per_gen=np.array(per_gen, dtype=object))
    return {"emb_all": emb_all, "gen_labels": gen_labels, "per_gen": list(per_gen)}


def main() -> None:
    ensure_dirs()
    set_science_style()
    benchmarks = ["make_moons", "halfcheetah"]
    results: dict[tuple[str, str], dict | None] = {}
    for b in benchmarks:
        wbg = load_weights(b, SEED)
        for m in METRICS:
            results[(b, m)] = get_embedding(b, m, wbg) if wbg is not None else None

    nrows, ncols = len(benchmarks), len(METRICS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.5 * nrows), dpi=130)
    axes = np.atleast_2d(axes)

    for r, b in enumerate(benchmarks):
        for c, m in enumerate(METRICS):
            ax = axes[r, c]
            ax.set_facecolor("white")
            payload = results[(b, m)]
            if payload is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", color="#a00")
            else:
                emb, gl = payload["emb_all"], payload["gen_labels"]
                ng = int(gl.max()) + 1 if len(gl) else 1
                ax.scatter(emb[:, 0], emb[:, 1], c=gl, cmap="plasma", vmin=0, vmax=max(ng - 1, 1),
                           s=5, alpha=0.6, edgecolors="none", rasterized=True)
                tc = temporal_coherence(payload["per_gen"])
                at = count_attractors_dbscan(emb, min_samples=5)
                ax.set_title(f"{m}\ntc={tc:.2f}  att={at}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(DISPLAY[b], fontsize=10)

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="plasma"), ax=axes[:, -1],
                        orientation="vertical", label="generation", shrink=0.6, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        f"UMAP metric comparison — benchmarks where evolution worked  "
        f"(nn=15, min_dist=0.1, λ={LAMBDA_ALIGN}, seed={SEED})", fontsize=13, y=1.01)
    fig.tight_layout()
    out = FIGURES_DIR / "exp12_metric_comparison_working.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")

    print("\ntc / attractors:")
    for b in benchmarks:
        for m in METRICS:
            p = results[(b, m)]
            if p is None:
                print(f"  {b:12s} {m:12s}  n/a"); continue
            tc = temporal_coherence(p["per_gen"]); at = count_attractors_dbscan(p["emb_all"], min_samples=5)
            print(f"  {b:12s} {m:12s}  tc={tc:.3f}  att={at}")


if __name__ == "__main__":
    main()
