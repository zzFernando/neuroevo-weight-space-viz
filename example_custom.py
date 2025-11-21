#!/usr/bin/env python3
"""
Example: Custom dataset with different parameters.

This example shows how to customize the genetic algorithm and neural network
for different problems.
"""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neuroevo.mlp import FixedMLP
from neuroevo.genetic_algorithm import GeneticAlgorithm


def main():
    """Run GA on the moons dataset with custom parameters."""
    print("=" * 80)
    print("Custom Example: Moons Dataset")
    print("=" * 80)
    
    # Create moons dataset (non-linear decision boundary)
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\nDataset: {len(X_train)} train, {len(X_test)} test samples")
    print(f"Features: {X_train.shape[1]}, Classes: {len(np.unique(y))}")
    
    # Smaller network with different parameters
    layer_sizes = [2, 8, 2]  # 2 inputs, 8 hidden, 2 outputs
    
    ga = GeneticAlgorithm(
        layer_sizes=layer_sizes,
        population_size=30,      # Smaller population
        mutation_rate=0.15,      # Higher mutation
        mutation_std=0.3,        # Lower mutation strength
        elite_size=3,
        tournament_size=5,       # Larger tournaments
        random_seed=42
    )
    
    print(f"\nGA Parameters:")
    print(f"  Population: {ga.population_size}")
    print(f"  Genome size: {ga.genome_size} weights")
    print(f"  Mutation rate: {ga.mutation_rate}")
    print(f"  Tournament size: {ga.tournament_size}")
    
    # Run for fewer generations
    print("\nEvolution:")
    history = ga.run(X_train, y_train, generations=50, verbose=True)
    
    # Evaluate on test set
    best_individual, _ = ga.get_best_individual()
    mlp = FixedMLP(layer_sizes)
    mlp.set_weights(best_individual)
    test_predictions = mlp.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test)
    
    print(f"\nFinal test accuracy: {test_accuracy:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
