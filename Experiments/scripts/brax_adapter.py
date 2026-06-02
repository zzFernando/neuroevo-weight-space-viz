"""
brax_adapter.py — Apply the paper's analysis pipeline to HalfCheetah .npz files.

Converts brax-format .npz (populations: G×P×D, fitnesses: G×P) to the paper's
EvolutionResult, then runs compute_aligned_umap_embedding + all Exp 1 metrics:
  - embedding spread σ1
  - temporal coherence (TC)
  - convergence rate
  - attractor count (DBSCAN)
  - exp1-style velocity-field figure

Usage:
    # From supplementary/
    pixi run python experiments/brax_adapter.py

    # Point at custom runs directory:
    pixi run python experiments/brax_adapter.py --runs_dir ../brax/runs

    # Force re-embed (ignore cache):
    pixi run python experiments/brax_adapter.py --force
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import compute_aligned_umap_embedding, EvolutionResult
from experiments.shared import (
    CACHE_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    ensure_dirs,
    count_attractors_dbscan,
    plot_compact_vector_field,
    save_results_csv,
    temporal_coherence,
    plot_fitness_panel,
)

logging.basicConfig(
    filename="errors.log", filemode="a",
    format="%(asctime)s %(levelname)s %(message)s", level=logging.ERROR,
)

SEEDS = [42, 123, 7, 31, 99]
LAMBDA_ALIGN = 0.8
GRID_RES = 22

# evosax algorithm names (must match neuroevolve_evosax.py --algo values)
EVOSAX_ALGOS = ["simple_ga", "open_es", "cma_es", "sep_cma_es", "snes", "xnes", "pepg"]


# ── Convergence rate (copied from 01_multi_seed_robustness.py) ────────────────

def _convergence_rate(mean_fitness: np.ndarray) -> int:
    start, final = float(mean_fitness[0]), float(mean_fitness[-1])
    span = final - start
    if abs(span) < 1e-10:
        return 1
    threshold = start + 0.95 * span
    reached = np.where(mean_fitness >= threshold)[0]
    return int(reached[0] + 1) if len(reached) else len(mean_fitness)


# ── Load & adapt brax .npz ────────────────────────────────────────────────────

def load_brax_npz(path: Path) -> EvolutionResult:
    """Convert brax-format .npz → EvolutionResult (paper's internal format)."""
    d = np.load(path, allow_pickle=True)
    populations = d["populations"]  # (G, P, D)
    fitnesses   = d["fitnesses"]    # (G, P)
    G, P, D = populations.shape

    weights_by_gen = [populations[g] for g in range(G)]   # list of (P, D)
    fitness_by_gen = [fitnesses[g]   for g in range(G)]   # list of (P,)

    mean_fitness = fitnesses.mean(axis=1)   # (G,)
    std_fitness  = fitnesses.std(axis=1)    # (G,)
    best_indices = fitnesses.argmax(axis=1) # (G,)

    return EvolutionResult(
        weights_by_gen=weights_by_gen,
        fitness_by_gen=fitness_by_gen,
        best_indices=best_indices,
        mean_fitness=mean_fitness,
        std_fitness=std_fitness,
    )


# ── Per-seed cache ────────────────────────────────────────────────────────────

def cache_path(seed: int, algo: str = "halfcheetah") -> Path:
    return CACHE_DIR / f"exp1_{algo}_{seed}.npz"


def get_cached_or_embed(npz_path: Path, seed: int, algo: str = "halfcheetah",
                        force: bool = False) -> dict | None:
    cp = cache_path(seed, algo)
    if cp.exists() and not force:
        try:
            data = np.load(cp, allow_pickle=True)
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

    print(f"  [seed {seed}] loading brax .npz…")
    result = load_brax_npz(npz_path)
    G = len(result.weights_by_gen)
    P = len(result.weights_by_gen[0])
    D = result.weights_by_gen[0].shape[1]
    print(f"  [seed {seed}] G={G} P={P} D={D}  →  running UMAP…")

    emb_all, gen_labels, per_gen = compute_aligned_umap_embedding(
        result.weights_by_gen,
        lambda_align=LAMBDA_ALIGN,
        random_state=seed,
        pca_dims=None,  # d_w=390: no PCA needed (paper §3 justification)
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cp,
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


# ── Figure ────────────────────────────────────────────────────────────────────

def build_figure(data_map: dict[int, dict | None], seeds: list[int],
                 algo: str = "halfcheetah") -> None:
    """3-row figure: velocity field / fitness / fitness curve."""
    n = len([s for s in seeds if data_map.get(s) is not None])
    if n == 0:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(3, n, figsize=(4.5 * n, 12), dpi=150)
    if n == 1:
        axes = axes[:, np.newaxis]

    col = 0
    for seed in seeds:
        payload = data_map.get(seed)
        if payload is None:
            continue

        # Row 0: velocity field (paper style)
        plot_compact_vector_field(
            axes[0, col], payload["per_gen"], payload["emb_all"], grid_res=GRID_RES
        )
        axes[0, col].set_title(f"seed {seed}", fontsize=9)
        if col == 0:
            axes[0, col].set_ylabel("Velocity field", fontsize=9)

        # Row 1: fitness coloring
        fitness_flat = np.concatenate(payload["mean_fitness"].reshape(-1, 1)
                                      .repeat(1, axis=1))
        # Use per-gen mean fitness broadcast to per-individual
        gen_labels = payload["gen_labels"]
        fitness_per_ind = payload["mean_fitness"][gen_labels]
        plot_fitness_panel(axes[1, col], payload["emb_all"], fitness_per_ind)
        if col == 0:
            axes[1, col].set_ylabel("By fitness", fontsize=9)

        # Row 2: fitness curve
        ax = axes[2, col]
        ax.plot(payload["mean_fitness"], color="#2196F3", linewidth=1.5, label="mean")
        ax.fill_between(
            range(len(payload["mean_fitness"])),
            payload["mean_fitness"] - payload["std_fitness"],
            payload["mean_fitness"] + payload["std_fitness"],
            alpha=0.2, color="#2196F3",
        )
        ax.set_xlabel("Generation", fontsize=8)
        if col == 0:
            ax.set_ylabel("Fitness", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_facecolor("white")

        col += 1

    fig.suptitle(
        f"HalfCheetah [{algo}] · {n} seeds · λ={LAMBDA_ALIGN} · paper methodology",
        fontsize=12, y=1.01,
    )
    # Colorbar for velocity field
    fig.colorbar(
        plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(0, 1)),
        ax=axes[0], orientation="vertical", label="Speed (norm.)",
        shrink=0.6, pad=0.01,
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"exp1_{algo}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _discover_runs(runs_dir: Path, seeds: list[int], algo: str) -> dict[int, Path]:
    """Find .npz files for a given algorithm and seed list."""
    patterns: list[str]
    if algo == "halfcheetah":
        # brax-native or stress-test data
        patterns = [f"halfcheetah_seed{{seed}}.npz", f"stress_seed{{seed}}.npz"]
    else:
        # evosax output: {algo}_seed{seed}.npz
        patterns = [f"{algo}_seed{{seed}}.npz"]

    found: dict[int, Path] = {}
    for seed in seeds:
        for pat in patterns:
            p = runs_dir / pat.format(seed=seed)
            if p.exists():
                found[seed] = p
                break
    return found


def main(runs_dir: Path, algo: str = "halfcheetah",
         force: bool = False, seeds: list[int] | None = None) -> None:
    ensure_dirs()
    if seeds is None:
        seeds = SEEDS

    available = _discover_runs(runs_dir, seeds, algo)

    if not available:
        print(f"No .npz files found for algo='{algo}' in {runs_dir}")
        if algo == "halfcheetah":
            print("Run: python brax/neuroevolve_brax.py --seed 42 --out brax/runs/halfcheetah_seed42.npz")
        else:
            print(f"Run: python brax/neuroevolve_evosax.py --algo {algo} --seed 42")
        return

    print(f"algo={algo}  found seeds: {sorted(available.keys())}")

    data_map: dict[int, dict | None] = {}
    metrics: list[dict] = []

    for seed in tqdm(sorted(available.keys()), desc=algo):
        try:
            payload = get_cached_or_embed(available[seed], seed, algo=algo, force=force)
            data_map[seed] = payload
            if payload is None:
                continue

            emb_all   = payload["emb_all"]
            per_gen   = payload["per_gen"]
            spread_1  = float(np.std(emb_all[:, 0]))
            spread_2  = float(np.std(emb_all[:, 1]))
            tc        = temporal_coherence(per_gen)
            attractors = count_attractors_dbscan(emb_all, min_samples=5)
            conv_rate  = _convergence_rate(payload["mean_fitness"])

            metrics.append({
                "benchmark":      "halfcheetah",
                "seed":           seed,
                "final_fitness":  float(payload["mean_fitness"][-1]),
                "spread_1":       round(spread_1, 4),
                "spread_2":       round(spread_2, 4),
                "temporal_coherence": round(tc, 4),
                "attractor_count": attractors,
                "convergence_rate": conv_rate,
            })

            print(
                f"  seed {seed}: σ1={spread_1:.3f}  TC={tc:.3f}"
                f"  conv_gen={conv_rate}  attractors={attractors}"
                f"  fitness={payload['mean_fitness'][-1]:.3f}"
            )
        except Exception as exc:
            logging.error("brax_adapter %s seed %s: %s", algo, seed, exc, exc_info=True)
            data_map[seed] = None
            print(f"  seed {seed}: FAILED — {exc}")

    if metrics:
        out_csv = RESULTS_DIR / f"exp1_{algo}_metrics.csv"
        save_results_csv(metrics, out_csv)
        import pandas as pd
        df = pd.DataFrame(metrics)
        print(f"\n── {algo} summary ──")
        print(df[["seed", "spread_1", "temporal_coherence",
                   "convergence_rate", "attractor_count", "final_fitness"]].to_string(index=False))
        print(f"\nMean σ1:  {df.spread_1.mean():.3f} ± {df.spread_1.std():.3f}")
        print(f"Mean TC:  {df.temporal_coherence.mean():.3f} ± {df.temporal_coherence.std():.3f}")
        print(f"Mean fitness: {df.final_fitness.mean():.3f} ± {df.final_fitness.std():.3f}")

    build_figure(data_map, sorted(available.keys()), algo=algo)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Apply paper analysis pipeline to brax HalfCheetah runs")
    p.add_argument("--runs_dir", type=Path,
                   default=Path(__file__).resolve().parents[2] / "brax" / "runs")
    p.add_argument("--algo",     default="halfcheetah",
                   help="Algorithm name: halfcheetah (brax GA) or evosax algo name "
                        f"({', '.join(EVOSAX_ALGOS)})")
    p.add_argument("--seeds",    type=int,  nargs="*", default=None)
    p.add_argument("--force",    action="store_true", help="Ignore embedding cache")
    args = p.parse_args()
    main(args.runs_dir, algo=args.algo, force=args.force, seeds=args.seeds)
