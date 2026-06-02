from __future__ import annotations

import argparse
import logging
import math
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


from benchmarks import DEFAULTS
from utils import compute_aligned_umap_embedding, run_evolution_benchmark
from shared import (
    CACHE_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    ensure_dirs,
    count_attractors_dbscan,
    plot_compact_vector_field,
    save_results_csv,
    temporal_coherence,
)

logging.basicConfig(
    filename="errors.log",
    filemode="a",
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.ERROR,
)

BENCHMARK = "make_moons"
SEED = 42

# Fixed UMAP params (paper-recommended); only the distance metric varies.
N_NEIGHBORS = 15
MIN_DIST = 0.1
LAMBDA_ALIGN = 0.8

# Metrics that require a special input domain (categorical labels, strings,
# lat/long pairs, simplex/probability vectors, graph embeddings). They are not
# meaningful on dense real-valued weight vectors, so we skip them up front.
SKIP_METRICS = {
    "categorical", "hierarchical_categorical", "ordinal", "count", "string",
    "haversine", "poincare", "ll_dirichlet",
}


def all_umap_metrics() -> list[str]:
    """Every named distance UMAP exposes, minus special-domain ones."""
    import umap.distances as ud

    names = sorted(ud.named_distances.keys())
    return [m for m in names if m not in SKIP_METRICS]


def metric_kwds_for(metric: str, all_weights: np.ndarray) -> dict | None:
    """Auto-derive required metric params from the data when possible."""
    if metric in ("minkowski", "wminkowski", "weighted_minkowski"):
        kwds: dict = {"p": 3}
        if metric in ("wminkowski", "weighted_minkowski"):
            kwds["w"] = np.ones(all_weights.shape[1], dtype=np.float64)
        return kwds
    if metric in ("seuclidean", "standardised_euclidean"):
        v = np.var(all_weights, axis=0)
        v[v == 0] = 1e-12
        return {"V": v.astype(np.float64)}
    if metric == "mahalanobis":
        cov = np.cov(all_weights, rowvar=False)
        vi = np.linalg.pinv(np.atleast_2d(cov))
        return {"VI": vi.astype(np.float64)}
    return None


def evo_cache_path() -> Path:
    return CACHE_DIR / f"exp8_evo_{BENCHMARK}_{SEED}.npz"


def get_evolution_data(force_rerun: bool = False) -> dict:
    path = evo_cache_path()
    if path.exists() and not force_rerun:
        try:
            data = np.load(path, allow_pickle=True)
            return {"weights_by_gen": list(data["weights_by_gen"])}
        except Exception:
            pass

    config = DEFAULTS[BENCHMARK].copy()
    result = run_evolution_benchmark(
        benchmark_name=BENCHMARK,
        pop_size=config["pop_size"],
        n_generations=config["n_generations"],
        hidden_dim=config["hidden_dim"],
        mutation_rate=config["mutation_rate"],
        seed=SEED,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        weights_by_gen=np.array(result.weights_by_gen, dtype=object),
    )
    return {"weights_by_gen": result.weights_by_gen}


def embed_cache_path(metric: str) -> Path:
    safe = metric.replace("-", "_")
    return CACHE_DIR / f"exp8_emb_{safe}.npz"


def get_embedding(
    weights_by_gen: list[np.ndarray],
    metric: str,
    metric_kwds: dict | None,
    force_rerun: bool = False,
) -> dict:
    path = embed_cache_path(metric)
    if path.exists() and not force_rerun:
        try:
            data = np.load(path, allow_pickle=True)
            return {
                "emb_all": data["emb_all"],
                "gen_labels": data["gen_labels"],
                "per_gen": list(data["per_gen"]),
            }
        except Exception:
            pass

    emb_all, gen_labels, per_gen = compute_aligned_umap_embedding(
        weights_by_gen,
        lambda_align=LAMBDA_ALIGN,
        random_state=SEED,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric=metric,
        metric_kwds=metric_kwds,
    )

    np.savez_compressed(
        path,
        emb_all=emb_all,
        gen_labels=gen_labels,
        per_gen=np.array(per_gen, dtype=object),
    )
    return {"emb_all": emb_all, "gen_labels": gen_labels, "per_gen": list(per_gen)}


def compute_metrics(emb_all: np.ndarray, per_gen: list[np.ndarray]) -> dict:
    return {
        "spread_1": float(np.std(emb_all[:, 0])),
        "spread_2": float(np.std(emb_all[:, 1])),
        "attractor_count": count_attractors_dbscan(emb_all, min_samples=5),
        "temporal_coherence": temporal_coherence(per_gen),
    }


def _plot_scatter_gen(ax: plt.Axes, emb_all: np.ndarray, gen_labels: np.ndarray):
    ax.set_facecolor("white")
    n_gens = int(gen_labels.max()) + 1 if len(gen_labels) else 1
    ax.scatter(
        emb_all[:, 0],
        emb_all[:, 1],
        c=gen_labels,
        cmap="plasma",
        vmin=0,
        vmax=max(n_gens - 1, 1),
        s=4,
        alpha=0.6,
        edgecolors="none",
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _auto_grid(n: int, ncols: int = 6) -> tuple[int, int]:
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def build_scatter_grid(metrics: list[str], results: dict) -> None:
    """One scatter (colored by generation) per metric, auto-laid out grid."""
    n = len(metrics)
    nrows, ncols = _auto_grid(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), dpi=120)
    axes = np.atleast_1d(axes).ravel()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        payload = results.get(metric)
        if payload is None:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=9, color="#a00")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(metric, fontsize=9)
            continue
        _plot_scatter_gen(ax, payload["emb_all"], payload["gen_labels"])
        tc = payload["metrics"]["temporal_coherence"]
        at = payload["metrics"]["attractor_count"]
        ax.set_title(f"{metric}\ntc={tc:.2f}  att={at}", fontsize=8.5)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"UMAP distance-metric comparison — scatter by generation  "
        f"(nn={N_NEIGHBORS}, min_dist={MIN_DIST}, λ={LAMBDA_ALIGN}; {BENCHMARK}, seed={SEED})",
        fontsize=12,
        y=1.005,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "exp8_metric_scatter_grid.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def build_vfield_grid(metrics: list[str], results: dict) -> None:
    """One vector field per metric, auto-laid out grid."""
    n = len(metrics)
    nrows, ncols = _auto_grid(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), dpi=120)
    axes = np.atleast_1d(axes).ravel()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        payload = results.get(metric)
        if payload is None:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=9, color="#a00")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(metric, fontsize=9)
            continue
        plot_compact_vector_field(ax, payload["per_gen"], payload["emb_all"], grid_res=16)
        ax.set_title(metric, fontsize=9)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"UMAP distance-metric comparison — evolutionary vector field  "
        f"(nn={N_NEIGHBORS}, min_dist={MIN_DIST}, λ={LAMBDA_ALIGN}; {BENCHMARK}, seed={SEED})",
        fontsize=12,
        y=1.005,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "exp8_metric_vfield_grid.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def main(force_rerun: bool = False) -> None:
    ensure_dirs()

    print(f"Loading / running evolution: {BENCHMARK}, seed={SEED}")
    evo = get_evolution_data(force_rerun=force_rerun)
    weights_by_gen = evo["weights_by_gen"]
    all_weights = np.vstack(list(weights_by_gen))

    metrics = all_umap_metrics()
    print(f"Testing {len(metrics)} metrics: {', '.join(metrics)}")

    results: dict = {}
    metrics_rows: list[dict] = []

    for metric in tqdm(metrics, desc="UMAP metrics"):
        try:
            kwds = metric_kwds_for(metric, all_weights)
            payload = get_embedding(weights_by_gen, metric, kwds, force_rerun=force_rerun)
            m = compute_metrics(payload["emb_all"], payload["per_gen"])
            payload["metrics"] = m
            results[metric] = payload

            metrics_rows.append({
                "benchmark": BENCHMARK,
                "seed": SEED,
                "metric": metric,
                "n_neighbors": N_NEIGHBORS,
                "min_dist": MIN_DIST,
                "lambda_align": LAMBDA_ALIGN,
                **m,
            })
        except Exception as exc:
            logging.error(
                "Exp8 failed metric=%s: %s\n%s", metric, exc, traceback.format_exc()
            )
            results[metric] = None

    save_results_csv(metrics_rows, RESULTS_DIR / "exp8_metric_comparison.csv")

    ok = [m for m in metrics if results.get(m) is not None]
    failed = [m for m in metrics if results.get(m) is None]

    if metrics_rows:
        df = pd.DataFrame(metrics_rows)
        print(f"\n{len(ok)}/{len(metrics)} metrics succeeded. Results by temporal coherence:")
        print(
            df[["metric", "temporal_coherence", "attractor_count", "spread_1", "spread_2"]]
            .sort_values("temporal_coherence", ascending=False)
            .to_string(index=False)
        )
    if failed:
        print(f"\nFailed/incompatible ({len(failed)}): {', '.join(failed)}")

    build_scatter_grid(metrics, results)
    build_vfield_grid(metrics, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 8: UMAP distance-metric comparison (all metrics)")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore cache and re-run")
    args = parser.parse_args()
    main(force_rerun=args.force_rerun)
