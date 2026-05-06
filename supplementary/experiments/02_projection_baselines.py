from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
)

logging.basicConfig(
    filename="errors.log",
    filemode="a",
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.ERROR,
)

SEED = 42
BENCHMARK_NAME = "make_moons"
GRID_RES = 22


def cache_path() -> Path:
    return CACHE_DIR / "exp2_make_moons_baselines.npz"


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
        logging.error("Exp2 evolution failed: %s\n%s", exc, traceback.format_exc())
        return None

    cache_payload = {
        "weights_by_gen": np.array(result.weights_by_gen, dtype=object),
        "fitness_by_gen": np.array(result.fitness_by_gen, dtype=object),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cache_payload)
    return cache_payload


def compute_baselines(weights_by_gen: list[np.ndarray]):
    all_weights = np.vstack(weights_by_gen)
    methods = {}

    pca_emb = PCA(n_components=2, random_state=SEED).fit_transform(all_weights)
    methods["PCA"] = {
        "emb_all": pca_emb,
        "per_gen": list(np.split(pca_emb, len(weights_by_gen))),
    }

    tsne_emb = TSNE(n_components=2, perplexity=30, random_state=SEED, init="random", learning_rate="auto").fit_transform(all_weights)
    methods["t-SNE"] = {
        "emb_all": tsne_emb,
        "per_gen": list(np.split(tsne_emb, len(weights_by_gen))),
    }

    emb_umap_unaligned, gen_labels, per_gen_unaligned = compute_aligned_umap_embedding(
        weights_by_gen,
        lambda_align=0.0,
        random_state=SEED,
        pca_dims=None,
    )
    emb_umap_aligned, _, per_gen_aligned = compute_aligned_umap_embedding(
        weights_by_gen,
        lambda_align=0.8,
        random_state=SEED,
        pca_dims=None,
    )

    methods["UMAP unaligned"] = {
        "emb_all": emb_umap_unaligned,
        "per_gen": per_gen_unaligned,
    }
    methods["UMAP aligned"] = {
        "emb_all": emb_umap_aligned,
        "per_gen": per_gen_aligned,
    }
    return methods


def build_figure(methods: dict[str, dict], fitness_concat: np.ndarray) -> None:
    method_names = list(methods.keys())
    n_methods = len(method_names)
    # Layout: 3 rows (gen, fitness, vf) × 4 cols (methods) — horizontal
    fig, axes = plt.subplots(3, n_methods, figsize=(18, 10), dpi=150)
    row_labels = ["Generation", "Fitness", "Vector field"]

    for col, method in enumerate(method_names):
        payload = methods[method]
        emb_all = payload["emb_all"]
        per_gen = payload["per_gen"]

        # Generation color
        ax = axes[0, col]
        ax.scatter(emb_all[:, 0], emb_all[:, 1],
                   c=np.repeat(np.arange(len(per_gen)), [len(x) for x in per_gen]),
                   cmap=plt.cm.turbo, s=8, alpha=0.8)
        ax.set_title(method, fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        # Fitness color
        ax = axes[1, col]
        plot_fitness_panel(ax, emb_all, fitness_concat)
        ax.set_xticks([])
        ax.set_yticks([])

        # Vector field
        ax = axes[2, col]
        plot_compact_vector_field(ax, per_gen, emb_all, grid_res=GRID_RES)
        ax.set_xticks([])
        ax.set_yticks([])

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=10)

    n_gens = len(methods[method_names[0]]["per_gen"])
    cb0 = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, n_gens - 1), cmap=plt.cm.turbo),
        ax=axes[0, :], orientation="vertical", label="Generation",
        shrink=0.8, pad=0.01, aspect=25,
    )
    cb0.set_ticks([0, n_gens // 2, n_gens - 1])
    cb0.ax.tick_params(labelsize=7)

    cb1 = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, 100), cmap="viridis"),
        ax=axes[1, :], orientation="vertical", label="Fitness (%)",
        shrink=0.8, pad=0.01, aspect=25,
    )
    cb1.set_ticks([0, 50, 100])
    cb1.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    cb1.ax.tick_params(labelsize=7)

    cb2 = fig.colorbar(
        plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(0, 1)),
        ax=axes[2, :], orientation="vertical", label="Speed (norm.)",
        shrink=0.8, pad=0.01, aspect=25,
    )
    cb2.set_ticks([0, 1])
    cb2.ax.tick_params(labelsize=7)

    fig_path = FIGURES_DIR / "exp2_projection_baselines.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {fig_path}")


def main(force_rerun: bool = False) -> None:
    ensure_dirs()
    data = get_cached_data(force_rerun=force_rerun)
    if data is None:
        print("Failed to obtain baseline data")
        return

    weights_by_gen = list(data["weights_by_gen"])
    fitness_by_gen = list(data["fitness_by_gen"])
    fitness_concat = np.concatenate(fitness_by_gen)

    methods = compute_baselines(weights_by_gen)
    metrics = []
    for method, payload in methods.items():
        emb_all = payload["emb_all"]
        modes = count_attractors_dbscan(emb_all, min_samples=5)
        metrics.append(
            {
                "method": method,
                "attractor_count": modes,
                "spread_1": float(np.std(emb_all[:, 0])),
                "spread_2": float(np.std(emb_all[:, 1])),
            }
        )

    save_results_csv(metrics, RESULTS_DIR / "exp2_projection_metrics.csv")
    build_figure(methods, fitness_concat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 2: Projection baselines")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore cache and re-run")
    args = parser.parse_args()
    main(force_rerun=args.force_rerun)
