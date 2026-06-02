"""Exp 10: side-by-side UMAP projections for the candidate metrics.

Visual companion to Exp 8/9: instead of summary statistics, this renders the
actual 2-D UMAP embeddings (scatter colored by generation) for each candidate
metric across a low-dim (make_moons) and a high-dim (cifar10) benchmark, so the
theory — angular metrics should help in high dimensions, L-metrics in low —
can be checked by eye. Reuses Exp 9 evolution caches.
"""
from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


from benchmarks import DEFAULTS
from utils import compute_aligned_umap_embedding, run_evolution_benchmark
from shared import (
    CACHE_DIR,
    FIGURES_DIR,
    ensure_dirs,
    count_attractors_dbscan,
    temporal_coherence,
)

logging.basicConfig(filename="errors.log", filemode="a",
                    format="%(asctime)s %(levelname)s %(message)s", level=logging.ERROR)

# Theoretical top-3 (mahalanobis, euclidean, cosine) + correlation (angular) +
# chebyshev (empirical contender), ordered for display.
METRICS = ["euclidean", "mahalanobis", "chebyshev", "cosine", "correlation"]
BENCHMARKS = ["make_moons", "cifar10"]
SEED = 42
PCA_DIMS: dict[str, int] = {}  # no PCA: UMAP runs directly on raw weight vectors
LAMBDA_ALIGN = 0.8
DISPLAY_NAMES = {"make_moons": "Make Moons (low-dim)", "cifar10": "CIFAR-10 (high-dim)"}

# mahalanobis needs a D×D inverse covariance; on raw high-dim weights (no PCA)
# that matrix is intractable (e.g. CIFAR-10 ~200k dims → ~300 GB). Skip above this.
MAX_MAHALANOBIS_DIM = 2000


def evo_cache_path(benchmark: str, seed: int) -> Path:
    # Reuse Exp 9 caches when present; fall back to an exp10-local copy.
    p9 = CACHE_DIR / f"exp9_evo_{benchmark}_{seed}.npz"
    return p9 if p9.exists() else CACHE_DIR / f"exp10_evo_{benchmark}_{seed}.npz"


def get_evolution_data(benchmark: str, seed: int) -> dict | None:
    path = evo_cache_path(benchmark, seed)
    if path.exists():
        try:
            data = np.load(path, allow_pickle=True)
            return {"weights_by_gen": list(data["weights_by_gen"])}
        except Exception:
            pass

    config = DEFAULTS[benchmark].copy()
    try:
        result = run_evolution_benchmark(
            benchmark_name=benchmark,
            pop_size=config["pop_size"],
            n_generations=config["n_generations"],
            hidden_dim=config["hidden_dim"],
            mutation_rate=config["mutation_rate"],
            seed=seed,
        )
    except Exception as exc:
        logging.error("Exp10 evolution failed %s: %s\n%s", benchmark, exc, traceback.format_exc())
        return None

    np.savez_compressed(CACHE_DIR / f"exp10_evo_{benchmark}_{seed}.npz",
                        weights_by_gen=np.array(result.weights_by_gen, dtype=object))
    return {"weights_by_gen": result.weights_by_gen}


def effective_features(weights_by_gen: list[np.ndarray], pca_dims: int | None, seed: int) -> np.ndarray:
    # Cached weights load back as dtype=object; cast so np.cov/PCA work.
    stacked = np.ascontiguousarray(np.vstack(list(weights_by_gen)), dtype=np.float64)
    if pca_dims is None:
        return stacked
    actual = min(pca_dims, stacked.shape[1], stacked.shape[0] - 1)
    return PCA(n_components=actual, random_state=seed).fit_transform(stacked)


def metric_kwds_for(metric: str, feats: np.ndarray) -> dict | None:
    if metric == "mahalanobis":
        vi = np.linalg.pinv(np.atleast_2d(np.cov(feats, rowvar=False)))
        return {"VI": vi.astype(np.float64)}
    return None


def get_embedding(benchmark: str, metric: str, wbg: list[np.ndarray], pca_dims: int | None) -> dict | None:
    # exp10-local cache (no PCA), kept separate from Exp 9's PCA-reduced embeddings.
    path = CACHE_DIR / f"exp10_emb_{benchmark}_{SEED}_{metric}.npz"
    if path.exists():
        try:
            data = np.load(path, allow_pickle=True)
            return {"emb_all": data["emb_all"], "gen_labels": data["gen_labels"], "per_gen": list(data["per_gen"])}
        except Exception:
            pass

    try:
        feats = effective_features(wbg, pca_dims, SEED)
        if metric == "mahalanobis" and feats.shape[1] > MAX_MAHALANOBIS_DIM:
            logging.error("Exp10 skip %s/mahalanobis: dim=%d too high for covariance (no PCA)",
                          benchmark, feats.shape[1])
            return None
        kwds = metric_kwds_for(metric, feats)
        emb_all, gen_labels, per_gen = compute_aligned_umap_embedding(
            wbg, lambda_align=LAMBDA_ALIGN, random_state=SEED,
            pca_dims=pca_dims, metric=metric, metric_kwds=kwds,
        )
    except Exception as exc:
        logging.error("Exp10 embed failed %s/%s: %s\n%s", benchmark, metric, exc, traceback.format_exc())
        return None

    np.savez_compressed(CACHE_DIR / f"exp10_emb_{benchmark}_{SEED}_{metric}.npz",
                        emb_all=emb_all, gen_labels=gen_labels, per_gen=np.array(per_gen, dtype=object))
    return {"emb_all": emb_all, "gen_labels": gen_labels, "per_gen": list(per_gen)}


def main() -> None:
    ensure_dirs()

    results: dict[tuple[str, str], dict | None] = {}
    for benchmark in BENCHMARKS:
        evo = get_evolution_data(benchmark, SEED)
        if evo is None:
            for m in METRICS:
                results[(benchmark, m)] = None
            continue
        pca_dims = PCA_DIMS.get(benchmark)
        for metric in METRICS:
            results[(benchmark, metric)] = get_embedding(benchmark, metric, evo["weights_by_gen"], pca_dims)

    nrows, ncols = len(BENCHMARKS), len(METRICS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.4 * nrows), dpi=130)
    axes = np.atleast_2d(axes)

    for r, benchmark in enumerate(BENCHMARKS):
        for c, metric in enumerate(METRICS):
            ax = axes[r, c]
            ax.set_facecolor("white")
            payload = results[(benchmark, metric)]
            if payload is None:
                msg = "intractable\n(D×D cov,\nno PCA)" if metric == "mahalanobis" else "n/a"
                ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=8, color="#a00")
                ax.set_title(metric, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                emb, gl = payload["emb_all"], payload["gen_labels"]
                n_gens = int(gl.max()) + 1 if len(gl) else 1
                ax.scatter(emb[:, 0], emb[:, 1], c=gl, cmap="plasma", vmin=0, vmax=max(n_gens - 1, 1),
                           s=5, alpha=0.6, edgecolors="none", rasterized=True)
                tc = temporal_coherence(payload["per_gen"])
                at = count_attractors_dbscan(emb, min_samples=5)
                ax.set_title(f"{metric}\ntc={tc:.2f}  att={at}", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(DISPLAY_NAMES[benchmark], fontsize=10)

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="plasma"), ax=axes[:, -1],
                        orientation="vertical", label="generation", shrink=0.6, pad=0.02)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"UMAP projections by distance metric — low-dim vs high-dim  "
        f"(nn=15, min_dist=0.1, λ={LAMBDA_ALIGN}, seed={SEED})",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "exp10_metric_projection_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
