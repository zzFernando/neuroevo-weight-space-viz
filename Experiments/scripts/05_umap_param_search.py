from __future__ import annotations

import argparse
import itertools
import logging
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import DEFAULTS
from utils import compute_aligned_umap_embedding, run_evolution_benchmark
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

BENCHMARK = "make_moons"
SEED = 42

N_NEIGHBORS_GRID = [5, 15, 30, 50]
MIN_DIST_GRID    = [0.0, 0.1, 0.3, 0.5]
LAMBDA_GRID      = [0.0, 0.3, 0.5, 0.8]


def evo_cache_path() -> Path:
    return CACHE_DIR / f"exp5_evo_{BENCHMARK}_{SEED}.npz"


def get_evolution_data(force_rerun: bool = False) -> dict:
    path = evo_cache_path()
    if path.exists() and not force_rerun:
        try:
            data = np.load(path, allow_pickle=True)
            return {
                "weights_by_gen": list(data["weights_by_gen"]),
                "mean_fitness": data["mean_fitness"],
            }
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
        mean_fitness=result.mean_fitness,
    )
    return {
        "weights_by_gen": result.weights_by_gen,
        "mean_fitness": result.mean_fitness,
    }


def embed_cache_path(n_neighbors: int, min_dist: float, lambda_align: float) -> Path:
    return CACHE_DIR / f"exp5_emb_nn{n_neighbors}_md{min_dist}_la{lambda_align}.npz"


def get_embedding(
    weights_by_gen: list[np.ndarray],
    n_neighbors: int,
    min_dist: float,
    lambda_align: float,
    force_rerun: bool = False,
) -> dict:
    path = embed_cache_path(n_neighbors, min_dist, lambda_align)
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
        lambda_align=lambda_align,
        random_state=SEED,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
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


def _plot_scatter_gen(
    ax: plt.Axes,
    emb_all: np.ndarray,
    gen_labels: np.ndarray,
) -> None:
    """Scatter plot colored by generation index."""
    ax.set_facecolor("white")
    n_gens = int(gen_labels.max()) + 1 if len(gen_labels) else 1
    sc = ax.scatter(
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
    return sc


def build_nn_minDist_figure(
    results: dict,
    lambda_align: float,
) -> None:
    """Grid: rows = n_neighbors, cols = min_dist*2 (scatter | vector field), fixed lambda_align."""
    n_rows = len(N_NEIGHBORS_GRID)
    n_md   = len(MIN_DIST_GRID)
    n_cols = n_md * 2  # scatter + vector field side by side

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 3.2 * n_rows),
        dpi=120,
    )

    for r, nn in enumerate(N_NEIGHBORS_GRID):
        for ci, md in enumerate(MIN_DIST_GRID):
            ax_sc  = axes[r, ci * 2]
            ax_vf  = axes[r, ci * 2 + 1]
            key    = (nn, md, lambda_align)
            payload = results.get(key)

            for ax in (ax_sc, ax_vf):
                if payload is None:
                    ax.text(0.5, 0.5, "failed", ha="center", va="center", fontsize=8, color="#a00")
                    ax.set_xticks([]); ax.set_yticks([])

            if payload is None:
                continue

            _plot_scatter_gen(ax_sc, payload["emb_all"], payload["gen_labels"])
            plot_compact_vector_field(ax_vf, payload["per_gen"], payload["emb_all"], grid_res=16)

            tc = payload["metrics"]["temporal_coherence"]
            at = payload["metrics"]["attractor_count"]
            ax_sc.set_title(f"tc={tc:.2f}  att={at}", fontsize=7.5)
            ax_vf.set_title("", fontsize=7.5)

            if ci == 0:
                ax_sc.set_ylabel(f"nn={nn}", fontsize=9)
            if r == 0:
                ax_sc.set_xlabel(f"min_dist={md}", fontsize=8.5)
                ax_sc.xaxis.set_label_position("top")
                ax_vf.set_xlabel("(vfield)", fontsize=7)
                ax_vf.xaxis.set_label_position("top")

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="plasma"),
        ax=axes[:, -1],
        orientation="vertical",
        label="generation",
        shrink=0.6,
        pad=0.02,
    )
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"UMAP param search — lambda_align={lambda_align}  ({BENCHMARK}, seed={SEED})",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()

    out = FIGURES_DIR / f"exp5_umap_grid_la{lambda_align}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def build_lambda_figure(results: dict) -> None:
    """Fixed best n_neighbors / min_dist, vary lambda_align — scatter + vector field."""
    nn_fixed = 15
    md_fixed = 0.1

    n_cols = len(LAMBDA_GRID)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.2 * n_cols, 7), dpi=120)

    for c, la in enumerate(LAMBDA_GRID):
        ax_sc = axes[0, c]
        ax_vf = axes[1, c]
        key   = (nn_fixed, md_fixed, la)
        payload = results.get(key)

        if payload is None:
            for ax in (ax_sc, ax_vf):
                ax.text(0.5, 0.5, "failed", ha="center", va="center", fontsize=9, color="#a00")
                ax.set_xticks([]); ax.set_yticks([])
            continue

        _plot_scatter_gen(ax_sc, payload["emb_all"], payload["gen_labels"])
        plot_compact_vector_field(ax_vf, payload["per_gen"], payload["emb_all"], grid_res=18)

        tc = payload["metrics"]["temporal_coherence"]
        ax_sc.set_title(f"λ={la}  tc={tc:.2f}", fontsize=9)
        ax_vf.set_title("vector field", fontsize=8)

    fig.suptitle(
        f"Lambda ablation — n_neighbors={nn_fixed}, min_dist={md_fixed}  ({BENCHMARK}, seed={SEED})",
        fontsize=11,
    )
    fig.tight_layout()

    out = FIGURES_DIR / "exp5_lambda_ablation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def main(force_rerun: bool = False) -> None:
    ensure_dirs()

    print(f"Loading / running evolution: {BENCHMARK}, seed={SEED}")
    evo = get_evolution_data(force_rerun=force_rerun)
    weights_by_gen = evo["weights_by_gen"]

    combos = list(itertools.product(N_NEIGHBORS_GRID, MIN_DIST_GRID, LAMBDA_GRID))
    results: dict = {}
    metrics_rows: list[dict] = []

    for nn, md, la in tqdm(combos, desc="UMAP combos"):
        try:
            payload = get_embedding(weights_by_gen, nn, md, la, force_rerun=force_rerun)
            m = compute_metrics(payload["emb_all"], payload["per_gen"])
            payload["metrics"] = m
            results[(nn, md, la)] = payload

            metrics_rows.append({
                "benchmark": BENCHMARK,
                "seed": SEED,
                "n_neighbors": nn,
                "min_dist": md,
                "lambda_align": la,
                **m,
            })
        except Exception as exc:
            logging.error(
                "Exp5 failed nn=%s md=%s la=%s: %s\n%s", nn, md, la, exc, traceback.format_exc()
            )
            results[(nn, md, la)] = None

    save_results_csv(metrics_rows, RESULTS_DIR / "exp5_umap_param_search.csv")

    df = pd.DataFrame(metrics_rows)

    # Best combos by temporal coherence
    top = df.nlargest(5, "temporal_coherence")[
        ["n_neighbors", "min_dist", "lambda_align", "temporal_coherence", "attractor_count"]
    ]
    print("\nTop-5 by temporal coherence:")
    print(top.to_string(index=False))

    # Summary: mean temporal_coherence per parameter, marginalised over others
    for param in ("n_neighbors", "min_dist", "lambda_align"):
        summary = df.groupby(param)["temporal_coherence"].mean().round(3)
        print(f"\ntemporal_coherence by {param}:\n{summary.to_string()}")

    for la in LAMBDA_GRID:
        build_nn_minDist_figure(results, lambda_align=la)

    build_lambda_figure(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 5: UMAP parameter search")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore cache and re-run")
    args = parser.parse_args()
    main(force_rerun=args.force_rerun)
