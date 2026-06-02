"""Exp 9: cross-benchmark / multi-seed validation of the top-3 UMAP metrics.

Exp 8 ranked distance metrics on a single benchmark/seed. Here we validate the
three legitimate candidates for signed weight-space — chebyshev, mahalanobis and
euclidean (control) — across all benchmarks and several seeds, to check whether
the ranking is stable rather than a one-off artifact.
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm


from benchmarks import DEFAULTS
from utils import compute_aligned_umap_embedding, run_evolution_benchmark
from shared import (
    CACHE_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    ensure_dirs,
    count_attractors_dbscan,
    save_results_csv,
    temporal_coherence,
)

logging.basicConfig(
    filename="errors.log",
    filemode="a",
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.ERROR,
)

METRICS = ["chebyshev", "mahalanobis", "euclidean"]
BENCHMARKS = ["make_moons", "cifar10", "halfcheetah"]
SEEDS = [42, 123, 7, 31, 99]
PCA_DIMS = {"cifar10": 50}
LAMBDA_ALIGN = 0.8
DISPLAY_NAMES = {"make_moons": "Make Moons", "cifar10": "CIFAR-10", "halfcheetah": "HalfCheetah"}


def evo_cache_path(benchmark: str, seed: int) -> Path:
    return CACHE_DIR / f"exp9_evo_{benchmark}_{seed}.npz"


def get_evolution_data(benchmark: str, seed: int, force_rerun: bool = False) -> dict | None:
    path = evo_cache_path(benchmark, seed)
    if path.exists() and not force_rerun:
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
        logging.error("Exp9 evolution failed %s seed %s: %s\n%s", benchmark, seed, exc, traceback.format_exc())
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, weights_by_gen=np.array(result.weights_by_gen, dtype=object))
    return {"weights_by_gen": result.weights_by_gen}


def effective_features(weights_by_gen: list[np.ndarray], pca_dims: int | None, seed: int) -> np.ndarray:
    """The stacked feature matrix UMAP actually sees (after optional PCA).

    Needed so mahalanobis' covariance is estimated in the same space as the
    embedding, avoiding a dimensionality mismatch with the PCA-reduced input.
    """
    stacked = np.vstack(list(weights_by_gen))
    if pca_dims is None:
        return stacked
    actual = min(pca_dims, stacked.shape[1], stacked.shape[0] - 1)
    return PCA(n_components=actual, random_state=seed).fit_transform(stacked)


def metric_kwds_for(metric: str, feats: np.ndarray) -> dict | None:
    if metric == "mahalanobis":
        cov = np.cov(feats, rowvar=False)
        vi = np.linalg.pinv(np.atleast_2d(cov))
        return {"VI": vi.astype(np.float64)}
    return None


def embed_cache_path(benchmark: str, seed: int, metric: str) -> Path:
    return CACHE_DIR / f"exp9_emb_{benchmark}_{seed}_{metric}.npz"


def get_embedding(
    benchmark: str,
    seed: int,
    metric: str,
    weights_by_gen: list[np.ndarray],
    pca_dims: int | None,
    force_rerun: bool = False,
) -> dict:
    path = embed_cache_path(benchmark, seed, metric)
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

    kwds = metric_kwds_for(metric, effective_features(weights_by_gen, pca_dims, seed))
    emb_all, gen_labels, per_gen = compute_aligned_umap_embedding(
        weights_by_gen,
        lambda_align=LAMBDA_ALIGN,
        random_state=seed,
        pca_dims=pca_dims,
        metric=metric,
        metric_kwds=kwds,
    )

    np.savez_compressed(
        path,
        emb_all=emb_all,
        gen_labels=gen_labels,
        per_gen=np.array(per_gen, dtype=object),
    )
    return {"emb_all": emb_all, "gen_labels": gen_labels, "per_gen": list(per_gen)}


def build_summary_figure(df: pd.DataFrame) -> None:
    """Bar chart: mean temporal coherence per metric, grouped by benchmark, ±std."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=130)

    x = np.arange(len(BENCHMARKS))
    width = 0.25
    colors = {"chebyshev": "#1f77b4", "mahalanobis": "#ff7f0e", "euclidean": "#2ca02c"}

    for panel, (col, ylabel) in enumerate(
        [("temporal_coherence", "temporal coherence"), ("attractor_count", "attractor count")]
    ):
        ax = axes[panel]
        for i, metric in enumerate(METRICS):
            means, stds = [], []
            for b in BENCHMARKS:
                sub = df[(df.metric == metric) & (df.benchmark == b)][col]
                means.append(sub.mean() if len(sub) else np.nan)
                stds.append(sub.std() if len(sub) > 1 else 0.0)
            ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3,
                   label=metric, color=colors[metric], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_NAMES[b] for b in BENCHMARKS])
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        if panel == 0:
            ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Top-3 UMAP metric validation — {len(SEEDS)} seeds × {len(BENCHMARKS)} benchmarks (λ={LAMBDA_ALIGN})",
        fontsize=12,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "exp9_metric_validation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def main(force_rerun: bool = False, skip: list[str] | None = None) -> None:
    ensure_dirs()
    skip = skip or []
    rows: list[dict] = []

    for benchmark in BENCHMARKS:
        if benchmark in skip:
            continue
        pca_dims = PCA_DIMS.get(benchmark)
        for seed in tqdm(SEEDS, desc=benchmark):
            if str(seed) in skip:
                continue
            evo = get_evolution_data(benchmark, seed, force_rerun=force_rerun)
            if evo is None:
                continue
            wbg = evo["weights_by_gen"]
            for metric in METRICS:
                try:
                    payload = get_embedding(benchmark, seed, metric, wbg, pca_dims, force_rerun=force_rerun)
                    rows.append({
                        "benchmark": benchmark,
                        "seed": seed,
                        "metric": metric,
                        "temporal_coherence": temporal_coherence(payload["per_gen"]),
                        "attractor_count": count_attractors_dbscan(payload["emb_all"], min_samples=5),
                        "spread_1": float(np.std(payload["emb_all"][:, 0])),
                        "spread_2": float(np.std(payload["emb_all"][:, 1])),
                    })
                except Exception as exc:
                    logging.error("Exp9 failed %s seed %s metric %s: %s\n%s",
                                  benchmark, seed, metric, exc, traceback.format_exc())

    if not rows:
        print("No results produced.")
        return

    df = pd.DataFrame(rows)
    save_results_csv(rows, RESULTS_DIR / "exp9_metric_validation.csv")

    # Aggregated: mean ± std per (benchmark, metric)
    agg = (
        df.groupby(["benchmark", "metric"])[["temporal_coherence", "attractor_count"]]
        .agg(["mean", "std"])
        .round(3)
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(RESULTS_DIR / "exp9_metric_validation_summary.csv", index=False)

    print("\nMean temporal coherence ± std per benchmark/metric:")
    for b in BENCHMARKS:
        sub = agg[agg.benchmark == b]
        if sub.empty:
            continue
        print(f"\n  {DISPLAY_NAMES[b]}:")
        for _, r in sub.sort_values("temporal_coherence_mean", ascending=False).iterrows():
            print(f"    {r['metric']:12s}  tc={r['temporal_coherence_mean']:.3f}±{r['temporal_coherence_std']:.3f}"
                  f"   att={r['attractor_count_mean']:.1f}±{r['attractor_count_std']:.1f}")

    # Overall mean across everything
    print("\nOverall (all benchmarks/seeds pooled), by temporal coherence:")
    overall = (
        df.groupby("metric")[["temporal_coherence", "attractor_count"]]
        .mean().round(3).sort_values("temporal_coherence", ascending=False)
    )
    print(overall.to_string())

    build_summary_figure(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 9: top-3 UMAP metric validation")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore cache and re-run")
    parser.add_argument("--skip", nargs="*", default=[], help="Benchmarks or seeds to skip")
    args = parser.parse_args()
    main(force_rerun=args.force_rerun, skip=[str(s) for s in args.skip])
