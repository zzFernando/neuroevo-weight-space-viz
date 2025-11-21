#!/usr/bin/env python3
"""
Main script for running genetic algorithm on neuroevolution and generating visualizations.

This script demonstrates:
1. A simple genetic algorithm that evolves only the weights of a fixed-topology MLP
2. Evaluation on a toy classification task (Iris dataset)
3. Logging of best individual and population each generation
4. MDS 2D trajectory visualization of best individual in weight space
5. UMAP 2D projections of population at start/mid/end, colored by fitness
"""
import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neuroevo.genetic_algorithm import GeneticAlgorithm
from neuroevo.visualizations import (
    plot_mds_trajectory,
    plot_umap_snapshots,
    plot_fitness_evolution
)


def create_toy_dataset(n_samples=500, n_features=4, n_classes=3, random_state=42):
    """
    Create a toy classification dataset.
    
    Args:
        n_samples: Number of samples
        n_features: Number of input features
        n_classes: Number of classes
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=1.0,
        random_state=random_state
    )
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def main():
    """Main execution function."""
    print("=" * 80)
    print("Neuroevolution Weight Space Visualization")
    print("=" * 80)
    print()
    
    # Create output directory for visualizations
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}/")
    print()
    
    # Create toy dataset
    print("Creating toy classification dataset...")
    n_features = 4
    n_classes = 3
    X_train, X_test, y_train, y_test = create_toy_dataset(
        n_samples=500,
        n_features=n_features,
        n_classes=n_classes,
        random_state=42
    )
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples:  {len(X_test)}")
    print(f"  Features:      {n_features}")
    print(f"  Classes:       {n_classes}")
    print()
    
    # Define MLP architecture
    layer_sizes = [n_features, 10, n_classes]  # Input, Hidden, Output
    print(f"MLP Architecture: {layer_sizes}")
    print()
    
    # Initialize genetic algorithm
    print("Initializing Genetic Algorithm...")
    ga = GeneticAlgorithm(
        layer_sizes=layer_sizes,
        population_size=50,
        mutation_rate=0.1,
        mutation_std=0.5,
        elite_size=5,
        tournament_size=3,
        random_seed=42
    )
    print(f"  Population size:    {ga.population_size}")
    print(f"  Genome size:        {ga.genome_size} weights")
    print(f"  Mutation rate:      {ga.mutation_rate}")
    print(f"  Elite size:         {ga.elite_size}")
    print()
    
    # Run genetic algorithm
    print("Running Genetic Algorithm...")
    print("-" * 80)
    n_generations = 100
    history = ga.run(X_train, y_train, generations=n_generations, verbose=True)
    print("-" * 80)
    print()
    
    # Evaluate best individual on test set
    best_individual, best_fitness_train = ga.get_best_individual()
    from neuroevo.mlp import FixedMLP
    mlp = FixedMLP(layer_sizes)
    mlp.set_weights(best_individual)
    predictions = mlp.predict(X_test)
    test_accuracy = np.mean(predictions == y_test)
    
    print("Final Results:")
    print(f"  Best train accuracy: {best_fitness_train:.4f}")
    print(f"  Test accuracy:       {test_accuracy:.4f}")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    print()
    
    # 1. Fitness evolution
    print("  1. Fitness evolution plot...")
    fig1 = plot_fitness_evolution(
        history,
        save_path=os.path.join(output_dir, "fitness_evolution.png")
    )
    
    # 2. MDS trajectory
    print("  2. MDS trajectory of best individual...")
    fig2 = plot_mds_trajectory(
        ga.best_individual_history,
        save_path=os.path.join(output_dir, "mds_trajectory.png")
    )
    
    # 3. UMAP snapshots
    print("  3. UMAP population snapshots...")
    fig3 = plot_umap_snapshots(
        ga.population_history,
        ga.fitness_history,
        snapshots=['start', 'mid', 'end'],
        save_path=os.path.join(output_dir, "umap_snapshots.png")
    )
    
    print()
    print("=" * 80)
    print("Done! Check the results/ directory for visualizations.")
    print("=" * 80)


if __name__ == "__main__":
    main()
