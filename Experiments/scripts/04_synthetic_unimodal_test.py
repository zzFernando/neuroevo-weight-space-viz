
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

from utils import compute_aligned_umap_embedding
from experiments.shared import (
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

SEED = 42
GRID_RES = 22

DATASETS = {
    "single_gaussian": {
        "description": "Single Gaussian cluster",
        "expected_modes": 1,
    },
    "four_gaussians": {
        "description": "4 Gaussian clusters",
        "expected_modes": 4,
    },
    "random_walk": {
        "description": "Random walk (no strong cluster structure)",
        "expected_modes": 0,
    },
}


def cache_path(dataset_name: str) -> Path:
    return CACHE_DIR / f"exp4_{dataset_name}.npz"


def generate_synthetic_sequence(dataset_name: str) -> list[np.ndarray]:
    rng = np.random.default_rng(SEED)
    n_points = 1000
    dim = 100
    if dataset_name == "single_gaussian":
        base = rng.normal(loc=0.0, scale=1.0, size=(n_points, dim))
        weights_by_gen = [base]
        for _ in range(1, 5):
            drift = rng.normal(scale=0.08, size=(n_points, dim))
            weights_by_gen.append(weights_by_gen[-1] + drift)
    elif dataset_name == "four_gaussians":
        centers = np.array([
            [5, 0] + [0] * (dim - 2),
            [-5, 0] + [0] * (dim - 2),
            [0, 5] + [0] * (dim - 2),
            [0, -5] + [0] * (dim - 2),
        ], dtype=float)
        points = []
        for center in centers:
            points.append(center + rng.normal(scale=0.8, size=(n_points // 4, dim)))
        base = np.vstack(points)
        weights_by_gen = [base]
        for _ in range(1, 5):
            drift = rng.normal(scale=0.08, size=(n_points, dim))
            weights_by_gen.append(weights_by_gen[-1] + drift)
    else:
        # random_walk: each generation independently sampled — no temporal structure
        weights_by_gen = []
        for _ in range(5):
            weights_by_gen.append(rng.normal(loc=0.0, scale=1.0, size=(n_points, dim)))
    return weights_by_gen


def get_cached_or_generate(dataset_name: str, force_rerun: bool = False) -> dict | None:
    path = cache_path(dataset_name)
    if path.exists() and not force_rerun:
        try:
            data = np.load(path, allow_pickle=True)
            return {
                "emb_all":          data["emb_all"],
                "gen_labels":       data["gen_labels"],
                "per_gen":          list(data["per_gen"]),
                "per_gen_unaligned": list(data["per_gen_unaligned"]),
            }
        except Exception:
            pass

    try:
        weights_by_gen = generate_synthetic_sequence(dataset_name)
        # pca_dims=10: reduces noise dimensions so UMAP can find cluster structure
        emb_all, gen_labels, per_gen = compute_aligned_umap_embedding(
            weights_by_gen,
            lambda_align=0.8,
            random_state=SEED,
            pca_dims=10,
        )
        # Unaligned embedding for coherence measurement (lambda=0 avoids artificially
        # inflated coherence caused by the alignment procedure itself)
        _, _, per_gen_unaligned = compute_aligned_umap_embedding(
            weights_by_gen,
            lambda_align=0.0,
            random_state=SEED,
            pca_dims=10,
        )
    except Exception as exc:
        logging.error("Synthetic generation failed for %s: %s\n%s", dataset_name, exc, traceback.format_exc())
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        weights_by_gen=np.array(weights_by_gen, dtype=object),
        emb_all=emb_all,
        gen_labels=gen_labels,
        per_gen=np.array(per_gen, dtype=object),
        per_gen_unaligned=np.array(per_gen_unaligned, dtype=object),
    )
    return {
        "weights_by_gen": weights_by_gen,
        "emb_all": emb_all,
        "gen_labels": gen_labels,
        "per_gen": per_gen,
        "per_gen_unaligned": per_gen_unaligned,
    }


def build_figure(results: dict[str, dict]) -> None:
    dataset_names = list(results.keys())
    n = len(dataset_names)
    # Layout: 2 rows (embedding, vector field) × 3 cols (datasets) — horizontal
    fig, axes = plt.subplots(2, n, figsize=(18, 8), dpi=150)
    row_labels = ["UMAP embedding", "Vector field"]

    for col, name in enumerate(dataset_names):
        payload = results[name]
        emb_all = payload["emb_all"]
        per_gen = payload["per_gen"]

        ax = axes[0, col]
        ax.scatter(emb_all[:, 0], emb_all[:, 1],
                   c=np.repeat(np.arange(len(per_gen)), [len(x) for x in per_gen]),
                   cmap=plt.cm.turbo, s=8, alpha=0.8)
        ax.set_title(name.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[1, col]
        plot_compact_vector_field(ax, per_gen, emb_all, grid_res=GRID_RES)
        ax.set_xticks([])
        ax.set_yticks([])

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=10)

    n_gens = len(results[dataset_names[0]]["per_gen"])
    cb0 = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, n_gens - 1), cmap=plt.cm.turbo),
        ax=axes[0, :], orientation="vertical", label="Generation",
        shrink=0.8, pad=0.01, aspect=25,
    )
    cb0.set_ticks(list(range(n_gens)))
    cb0.ax.tick_params(labelsize=7)

    cb1 = fig.colorbar(
        plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(0, 1)),
        ax=axes[1, :], orientation="vertical", label="Speed (norm.)",
        shrink=0.8, pad=0.01, aspect=25,
    )
    cb1.set_ticks([0, 1])
    cb1.ax.tick_params(labelsize=7)

    fig_path = FIGURES_DIR / "exp4_synthetic_validation.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {fig_path}")


def main(force_rerun: bool = False) -> None:
    ensure_dirs()
    records = []
    results = {}

    for dataset_name in tqdm(DATASETS, desc="synthetic datasets"):
        payload = get_cached_or_generate(dataset_name, force_rerun=force_rerun)
        if payload is None:
            continue

        emb_all = payload["emb_all"]
        per_gen = payload["per_gen"]
        # Use unaligned embeddings for coherence: aligned embeddings have artificially
        # high coherence (lambda forces 80% overlap with reference), masking differences
        coherence = temporal_coherence(payload["per_gen_unaligned"])
        attractors = count_attractors_dbscan(emb_all, min_samples=5)
        spread_1 = float(np.std(emb_all[:, 0]))
        spread_2 = float(np.std(emb_all[:, 1]))

        records.append(
            {
                "dataset": dataset_name,
                "expected_modes": DATASETS[dataset_name]["expected_modes"],
                "attractor_count": attractors,
                "spread_1": spread_1,
                "spread_2": spread_2,
                "temporal_coherence": coherence,
            }
        )
        results[dataset_name] = payload

    save_results_csv(records, RESULTS_DIR / "exp4_synthetic_validation.csv")
    build_figure(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 4: Synthetic unimodal validation")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore cache and re-run")
    args = parser.parse_args()
    main(force_rerun=args.force_rerun)
