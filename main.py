#!/usr/bin/env python3
"""Entry point for running the streamlined neuroevolution demo."""
from __future__ import annotations

import os
import argparse
import numpy as np

from neuroevo.datasets import generate_complex_dataset
from neuroevo.mlp import FixedMLP
from neuroevo.genetic_algorithm import GeneticAlgorithm
from neuroevo.visualizations import (
    plot_weight_heatmap,
    plot_best_mds,
    plot_population_weight_stats,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_experiment(args: argparse.Namespace) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = generate_complex_dataset(
        n_samples=args.samples,
        seed=args.seed,
        n_classes=args.classes
    )

    layer_sizes = [X_train.shape[1], 64, 32, y_train.max() + 1]
    print(f"Dataset: {X_train.shape[0]} train / {X_test.shape[0]} test")
    print(f"Features: {X_train.shape[1]} | Classes: {y_train.max() + 1}")
    print(f"MLP architecture: {layer_sizes}")

    ga = GeneticAlgorithm(
        layer_sizes=layer_sizes,
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        mutation_std=args.mutation_std,
        elite_size=args.elite_size,
        tournament_size=args.tournament_size,
        random_seed=args.seed,
    )

    history = ga.run(X_train, y_train, generations=args.generations, verbose=True)

    best_payload = ga.get_best_individual()
    mlp = FixedMLP(layer_sizes)
    mlp.set_weights(best_payload["weights"])
    preds = mlp.predict(X_test)
    test_accuracy = np.mean(preds == y_test)
    print(f"\nBest train accuracy: {best_payload['fitness']:.3f}")
    print(f"Test accuracy:       {test_accuracy:.3f}")

    print("\nGenerating visualizations...")
    plot_weight_heatmap(
        ga.best_individual_history,
        os.path.join(RESULTS_DIR, "weights_heatmap.png")
    )
    plot_best_mds(
        ga.best_individual_history,
        os.path.join(RESULTS_DIR, "weights_mds.png")
    )
    plot_population_weight_stats(
        ga.best_individual_history,
        ga.population_history,
        os.path.join(RESULTS_DIR, "population_weight_stats.png")
    )
    print(f"Artifacts saved to {RESULTS_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neuroevolution visualization demo")
    parser.add_argument("--generations", type=int, default=180)
    parser.add_argument("--population-size", type=int, default=70)
    parser.add_argument("--mutation-rate", type=float, default=0.12)
    parser.add_argument("--mutation-std", type=float, default=0.35)
    parser.add_argument("--elite-size", type=int, default=5)
    parser.add_argument("--tournament-size", type=int, default=4)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
