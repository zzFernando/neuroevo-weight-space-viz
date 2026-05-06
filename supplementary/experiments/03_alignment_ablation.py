

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import compute_aligned_umap_embedding, run_evolution_benchmark
from experiments.shared import (
    CACHE_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    ensure_dirs,
    count_attractors_dbscan,
    plot_compact_vector_field,
    plot_fitness_panel,
    save_results_csv,
    temporal_coherence,
)

logging.basicConfig(
    filename="errors.log",
    filemode="a",
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.ERROR,
)

LAMBDA_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SEED = 42
BENCHMARK_NAME = "make_moons"
GRID_RES = 22


def cache_path() -> Path:
    return CACHE_DIR / "exp3_alignment_ablation.npz"


def get_cached_data(force_rerun: bool = False) -> dict | None:
    path = cache_path()
    if path.exists() and not force_rerun:
        try:
            data = np.load(path, allow_pickle=True)
            return {key: data[key] for key in data.files}
        except Exception:
            pass

    try:
        result = run_evolution_benchmark(
            benchmark_name=BENCHMARK_NAME,
            pop_size=50,
            n_generations=80,
            hidden_dim=16,
            mutation_rate=0.11,
            seed=SEED,
        )
    except Exception as exc:
        logging.error("Exp3 evolution failed: %s\n%s", exc, traceback.format_exc())
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        weights_by_gen=np.array(result.weights_by_gen, dtype=object),
        fitness_by_gen=np.array(result.fitness_by_gen, dtype=object),
    )
    return {"weights_by_gen": result.weights_by_gen, "fitness_by_gen": result.fitness_by_gen}


def build_figure(results: list[dict]) -> None:
    # Main grid: 3 rows (gen, fitness, vf) × 6 cols (lambda values)
    fig = plt.figure(figsize=(24, 9), dpi=150)
    gs = fig.add_gridspec(3, len(LAMBDA_SWEEP), hspace=0.30, wspace=0.30)

    row_labels = ["Generation", "Fitness", "Vector field"]
    axes_grid: dict[tuple[int, int], plt.Axes] = {}

    for col, result in enumerate(results):
        lam = result["lambda_align"]
        emb_all = result["emb_all"]
        per_gen = result["per_gen"]
        fitness_concat = result["fitness_concat"]

        # Clip axes to 1–99th percentile so extreme λ values don't blow the scale
        xlo, xhi = np.percentile(emb_all[:, 0], [1, 99])
        ylo, yhi = np.percentile(emb_all[:, 1], [1, 99])
        pad_x = max((xhi - xlo) * 0.05, 1e-3)
        pad_y = max((yhi - ylo) * 0.05, 1e-3)
        xlim = (xlo - pad_x, xhi + pad_x)
        ylim = (ylo - pad_y, yhi + pad_y)

        ax = fig.add_subplot(gs[0, col])
        axes_grid[(0, col)] = ax
        ax.scatter(emb_all[:, 0], emb_all[:, 1],
                   c=np.repeat(np.arange(len(per_gen)), [len(x) for x in per_gen]),
                   cmap=plt.cm.turbo, s=8, alpha=0.8)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"$\\lambda={lam}$", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(row_labels[0], fontsize=10)

        ax = fig.add_subplot(gs[1, col])
        axes_grid[(1, col)] = ax
        plot_fitness_panel(ax, emb_all, fitness_concat)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(row_labels[1], fontsize=10)

        ax = fig.add_subplot(gs[2, col])
        axes_grid[(2, col)] = ax
        plot_compact_vector_field(ax, per_gen, emb_all, grid_res=GRID_RES)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(row_labels[2], fontsize=10)

    n_cols = len(LAMBDA_SWEEP)
    n_gens = len(results[0]["per_gen"])
    fitness_concat_sample = results[0]["fitness_concat"]
    axes_row0 = [axes_grid[(0, c)] for c in range(n_cols)]
    axes_row1 = [axes_grid[(1, c)] for c in range(n_cols)]
    axes_row2 = [axes_grid[(2, c)] for c in range(n_cols)]

    cb0 = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, n_gens - 1), cmap=plt.cm.turbo),
        ax=axes_row0, orientation="vertical", label="Generation",
        shrink=0.8, pad=0.01, aspect=25,
    )
    cb0.set_ticks([0, n_gens // 2, n_gens - 1])
    cb0.ax.tick_params(labelsize=7)

    cb1 = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, 100), cmap="viridis"),
        ax=axes_row1, orientation="vertical", label="Fitness (%)",
        shrink=0.8, pad=0.01, aspect=25,
    )
    cb1.set_ticks([0, 50, 100])
    cb1.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    cb1.ax.tick_params(labelsize=7)

    cb2 = fig.colorbar(
        plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(0, 1)),
        ax=axes_row2, orientation="vertical", label="Speed (norm.)",
        shrink=0.8, pad=0.01, aspect=25,
    )
    cb2.set_ticks([0, 1])
    cb2.ax.tick_params(labelsize=7)
    fig_path = FIGURES_DIR / "exp3_alignment_ablation.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {fig_path}")

    # Standalone TC vs lambda chart
    fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=150)
    x = [r["lambda_align"] for r in results]
    y = [r["temporal_coherence"] for r in results]
    ax2.plot(x, y, marker="o", color="#264653", linewidth=2, markersize=8)
    ax2.axvline(0.8, color="#e76f51", linestyle="--", linewidth=1.5, label="$\\lambda=0.8$ (selected)")
    ax2.set_xlabel("$\\lambda$", fontsize=13)
    ax2.set_ylabel("Temporal coherence (TC)", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_ylim(-0.1, 1.05)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, linestyle="--")
    fig2.tight_layout()
    fig2_path = FIGURES_DIR / "exp3_tc_curve.png"
    fig2.savefig(fig2_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"Saved {fig2_path}")


def main(force_rerun: bool = False) -> None:
    ensure_dirs()
    data = get_cached_data(force_rerun=force_rerun)
    if data is None:
        print("Failed to obtain data for Exp 3")
        return

    weights_by_gen = list(data["weights_by_gen"])
    fitness_by_gen = list(data["fitness_by_gen"])
    fitness_concat = np.concatenate(fitness_by_gen)
    results = []

    for lam in tqdm(LAMBDA_SWEEP, desc="lambda sweep"):
        emb_all, gen_labels, per_gen = compute_aligned_umap_embedding(
            weights_by_gen,
            lambda_align=lam,
            random_state=SEED,
            pca_dims=None,
        )
        coherence = temporal_coherence(per_gen)
        attractors = count_attractors_dbscan(emb_all, min_samples=5)
        results.append(
            {
                "lambda_align": lam,
                "temporal_coherence": coherence,
                "attractor_count": attractors,
                "spread_1": float(np.std(emb_all[:, 0])),
                "spread_2": float(np.std(emb_all[:, 1])),
                "emb_all": emb_all,
                "per_gen": per_gen,
                "fitness_concat": fitness_concat,
            }
        )

    output_metrics = [
        {
            "lambda_align": r["lambda_align"],
            "temporal_coherence": r["temporal_coherence"],
            "attractor_count": r["attractor_count"],
            "spread_1": r["spread_1"],
            "spread_2": r["spread_2"],
        }
        for r in results
    ]
    save_results_csv(output_metrics, RESULTS_DIR / "exp3_alignment_coherence.csv")
    build_figure(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 3: Alignment ablation")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore cache and re-run")
    args = parser.parse_args()
    main(force_rerun=args.force_rerun)
