
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

BENCHMARKS = ["make_moons", "cifar10", "halfcheetah"]
SEEDS = [42, 123, 7, 31, 99]
PCA_DIMS = {"cifar10": 50}
GRID_RES = 22
DISPLAY_NAMES = {"make_moons": "Make Moons", "cifar10": "CIFAR-10", "halfcheetah": "HalfCheetah"}


def cache_path(benchmark_name: str, seed: int) -> Path:
    return CACHE_DIR / f"exp1_{benchmark_name}_{seed}.npz"


def get_cached_or_run(benchmark_name: str, seed: int, force_rerun: bool = False) -> dict | None:
    path = cache_path(benchmark_name, seed)
    if path.exists() and not force_rerun:
        try:
            data = np.load(path, allow_pickle=True)
            return {
                "mean_fitness": data["mean_fitness"],
                "std_fitness":  data["std_fitness"],
                "best_indices": data["best_indices"],
                "emb_all":      data["emb_all"],
                "gen_labels":   data["gen_labels"],
                "per_gen":      list(data["per_gen"]),
            }
        except Exception:
            pass

    config = DEFAULTS[benchmark_name].copy()
    if benchmark_name in PCA_DIMS:
        config["pca_dims"] = PCA_DIMS[benchmark_name]
    else:
        config["pca_dims"] = None

    try:
        result = run_evolution_benchmark(
            benchmark_name=benchmark_name,
            pop_size=config["pop_size"],
            n_generations=config["n_generations"],
            hidden_dim=config["hidden_dim"],
            mutation_rate=config["mutation_rate"],
            seed=seed,
        )
    except Exception as exc:
        logging.error("Evolution failed for %s seed %s: %s\n%s", benchmark_name, seed, exc, traceback.format_exc())
        return None

    emb_all, gen_labels, per_gen = compute_aligned_umap_embedding(
        result.weights_by_gen,
        lambda_align=0.8,
        random_state=seed,
        pca_dims=config["pca_dims"],
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # weights_by_gen and fitness_by_gen excluded: too large for high-dim benchmarks
    # (CIFAR-10: ~1.2 GB/seed). Only metrics and embeddings are needed for analysis.
    np.savez_compressed(
        path,
        mean_fitness=result.mean_fitness,
        std_fitness=result.std_fitness,
        best_indices=result.best_indices,
        emb_all=emb_all,
        gen_labels=gen_labels,
        per_gen=np.array(per_gen, dtype=object),
    )
    return {
        "mean_fitness": result.mean_fitness,
        "std_fitness":  result.std_fitness,
        "best_indices": result.best_indices,
        "emb_all":      emb_all,
        "gen_labels":   gen_labels,
        "per_gen":      per_gen,
    }


def compute_convergence_rate(mean_fitness: np.ndarray) -> int:
    if len(mean_fitness) == 0:
        return -1
    start = float(mean_fitness[0])
    final = float(mean_fitness[-1])
    span = final - start
    if abs(span) < 1e-10:
        return 1
    threshold = start + 0.95 * span
    reached = np.where(mean_fitness >= threshold)[0]
    return int(reached[0] + 1) if len(reached) else len(mean_fitness)


def build_figure(data_map: dict[str, dict | None]) -> None:
    fig, axes = plt.subplots(len(BENCHMARKS), len(SEEDS), figsize=(20, 8), dpi=150)
    for row, benchmark_name in enumerate(BENCHMARKS):
        for col, seed in enumerate(SEEDS):
            ax = axes[row, col]
            payload = data_map[benchmark_name].get(seed)
            if payload is None:
                ax.text(0.5, 0.5, "failed", ha="center", va="center", fontsize=9, color="#a00")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            plot_compact_vector_field(ax, payload["per_gen"], payload["emb_all"], grid_res=GRID_RES)
            if row == 0:
                ax.set_title(f"seed {seed}", fontsize=9)
            if col == 0:
                ax.set_ylabel(DISPLAY_NAMES.get(benchmark_name, benchmark_name), fontsize=10)

    cb = fig.colorbar(
        plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(0, 1)),
        ax=axes, orientation="vertical", label="Speed (norm.)",
        shrink=0.5, pad=0.01, aspect=30,
    )
    cb.set_ticks([0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=8)

    fig.savefig(FIGURES_DIR / "exp1_multiseed.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / 'exp1_multiseed.png'}")


def main(force_rerun: bool = False, skip: list[str] | None = None) -> None:
    ensure_dirs()
    skip = skip or []
    metrics: list[dict[str, object]] = []
    data_map: dict[str, dict[int, dict] | None] = {name: {} for name in BENCHMARKS}

    for benchmark_name in BENCHMARKS:
        if benchmark_name in skip:
            continue
        data_map[benchmark_name] = {}
        for seed in tqdm(SEEDS, desc=f"{benchmark_name}"):
            if str(seed) in skip:
                continue
            try:
                payload = get_cached_or_run(benchmark_name, seed, force_rerun=force_rerun)
                data_map[benchmark_name][seed] = payload
                if payload is None:
                    continue
                emb_all = payload["emb_all"]
                final_fitness = float(payload["mean_fitness"][-1])
                spread_1 = float(np.std(emb_all[:, 0]))
                spread_2 = float(np.std(emb_all[:, 1]))
                attractors = count_attractors_dbscan(emb_all, min_samples=5)
                convergence = compute_convergence_rate(payload["mean_fitness"])

                metrics.append(
                    {
                        "benchmark": benchmark_name,
                        "seed": seed,
                        "final_fitness": final_fitness,
                        "spread_1": spread_1,
                        "spread_2": spread_2,
                        "attractor_count": attractors,
                        "convergence_rate": convergence,
                    }
                )
            except Exception as exc:
                logging.error("Exp1 failed for %s seed %s: %s\n%s", benchmark_name, seed, exc, traceback.format_exc())
                data_map[benchmark_name][seed] = None

    save_results_csv(metrics, RESULTS_DIR / "exp1_multiseed_metrics.csv")

    import pandas as pd
    from scipy import stats

    df = pd.DataFrame(metrics)

    # Inter-seed robustness summary
    summary = (
        df.groupby("benchmark")[["final_fitness", "spread_1", "attractor_count", "convergence_rate"]]
        .agg(["mean", "std"])
        .round(3)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(RESULTS_DIR / "exp1_multiseed_summary.csv", index=False)

    # Statistical tests: Welch's t-test + Cohen's d between benchmarks
    stat_rows = []
    pairs = [
        ("make_moons", "cifar10", "spread_1"),
        ("make_moons", "cifar10", "convergence_rate"),
        ("make_moons", "cifar10", "final_fitness"),
    ]
    for b1, b2, metric in pairs:
        a = df[df.benchmark == b1][metric].values
        b = df[df.benchmark == b2][metric].values
        if len(a) < 2 or len(b) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)  # Welch's t-test
        pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
        cohens_d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else float("nan")
        df_welch = (np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)) ** 2 / (
            (np.var(a, ddof=1) / len(a)) ** 2 / (len(a) - 1)
            + (np.var(b, ddof=1) / len(b)) ** 2 / (len(b) - 1)
        )
        stat_rows.append({
            "metric": metric,
            "benchmark_a": b1,
            "benchmark_b": b2,
            "mean_a": round(float(np.mean(a)), 4),
            "mean_b": round(float(np.mean(b)), 4),
            "std_a": round(float(np.std(a, ddof=1)), 4),
            "std_b": round(float(np.std(b, ddof=1)), 4),
            "t_stat": round(float(t_stat), 4),
            "df_welch": round(float(df_welch), 2),
            "p_value": round(float(p_val), 6),
            "cohens_d": round(float(cohens_d), 3),
            "n_a": len(a),
            "n_b": len(b),
        })
        print(
            f"  {metric:20s}  t({df_welch:.1f})={t_stat:+.3f}  "
            f"p={p_val:.4f}  d={cohens_d:.2f}"
        )

    pd.DataFrame(stat_rows).to_csv(RESULTS_DIR / "exp1_statistical_tests.csv", index=False)

    build_figure(data_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 1: Multi-seed robustness")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore cache and re-run experiments")
    parser.add_argument("--skip", nargs="*", default=[], help="Benchmarks or seeds to skip, e.g. cifar10 42")
    args = parser.parse_args()

    skip_targets = [str(item) for item in args.skip]
    main(force_rerun=args.force_rerun, skip=skip_targets)
