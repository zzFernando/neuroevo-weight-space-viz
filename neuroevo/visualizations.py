"""Visualization tools for weight space analysis."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from umap import UMAP


def plot_mds_trajectory(best_individual_history, save_path=None):
    """
    Plot MDS 2D trajectory of the best individual over generations.
    
    Args:
        best_individual_history: List of best individuals (weight vectors) per generation
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib figure object
    """
    # Convert to array
    weight_matrix = np.array(best_individual_history)
    n_generations = len(weight_matrix)
    
    # Apply MDS to reduce to 2D
    mds = MDS(n_components=2, random_state=42, dissimilarity='euclidean')
    weights_2d = mds.fit_transform(weight_matrix)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot trajectory as a line
    ax.plot(weights_2d[:, 0], weights_2d[:, 1], 'b-', alpha=0.3, linewidth=1)
    
    # Plot points colored by generation
    scatter = ax.scatter(weights_2d[:, 0], weights_2d[:, 1], 
                        c=np.arange(n_generations), 
                        cmap='viridis', 
                        s=50, 
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=0.5)
    
    # Mark start and end
    ax.scatter(weights_2d[0, 0], weights_2d[0, 1], 
              c='green', s=200, marker='*', 
              edgecolors='black', linewidth=1.5,
              label='Start', zorder=5)
    ax.scatter(weights_2d[-1, 0], weights_2d[-1, 1], 
              c='red', s=200, marker='*', 
              edgecolors='black', linewidth=1.5,
              label='End', zorder=5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Generation', rotation=270, labelpad=20)
    
    ax.set_xlabel('MDS Dimension 1', fontsize=12)
    ax.set_ylabel('MDS Dimension 2', fontsize=12)
    ax.set_title('MDS 2D Trajectory of Best Individual in Weight Space', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"MDS trajectory plot saved to {save_path}")
    
    return fig


def plot_umap_snapshots(population_history, fitness_history, 
                        snapshots=['start', 'mid', 'end'],
                        save_path=None):
    """
    Plot UMAP 2D projections of population at different time points.
    
    Args:
        population_history: List of population arrays per generation
        fitness_history: List of fitness arrays per generation
        snapshots: List of snapshot names ('start', 'mid', 'end') or generation indices
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib figure object
    """
    n_generations = len(population_history)
    
    # Determine snapshot indices
    snapshot_indices = []
    snapshot_labels = []
    
    for snap in snapshots:
        if isinstance(snap, str):
            if snap == 'start':
                idx = 0
                label = 'Start (Gen 0)'
            elif snap == 'mid':
                idx = n_generations // 2
                label = f'Mid (Gen {idx})'
            elif snap == 'end':
                idx = n_generations - 1
                label = f'End (Gen {idx})'
            else:
                raise ValueError(f"Unknown snapshot name: {snap}")
        else:
            idx = snap
            label = f'Gen {idx}'
        
        snapshot_indices.append(idx)
        snapshot_labels.append(label)
    
    # Create subplots
    n_snapshots = len(snapshot_indices)
    fig, axes = plt.subplots(1, n_snapshots, figsize=(7*n_snapshots, 6))
    
    if n_snapshots == 1:
        axes = [axes]
    
    for ax, idx, label in zip(axes, snapshot_indices, snapshot_labels):
        population = population_history[idx]
        fitness = fitness_history[idx]
        
        # Apply UMAP
        umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        pop_2d = umap_model.fit_transform(population)
        
        # Plot with fitness as color
        scatter = ax.scatter(pop_2d[:, 0], pop_2d[:, 1], 
                           c=fitness, 
                           cmap='RdYlGn',
                           s=100, 
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5,
                           vmin=0.0,
                           vmax=1.0)
        
        # Mark best individual
        best_idx = np.argmax(fitness)
        ax.scatter(pop_2d[best_idx, 0], pop_2d[best_idx, 1],
                  c='blue', s=300, marker='*',
                  edgecolors='black', linewidth=2,
                  label=f'Best (fit={fitness[best_idx]:.3f})',
                  zorder=5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Fitness', rotation=270, labelpad=20)
        
        ax.set_xlabel('UMAP Dimension 1', fontsize=11)
        ax.set_ylabel('UMAP Dimension 2', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('UMAP 2D Population Projections: Diversity and Convergence', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"UMAP snapshots plot saved to {save_path}")
    
    return fig


def plot_fitness_evolution(history, save_path=None):
    """
    Plot fitness evolution over generations.
    
    Args:
        history: Dictionary with 'generation', 'best_fitness', 'mean_fitness', 'std_fitness'
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    generations = history['generation']
    best_fitness = history['best_fitness']
    mean_fitness = history['mean_fitness']
    std_fitness = history['std_fitness']
    
    # Plot best and mean fitness
    ax.plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness')
    ax.plot(generations, mean_fitness, 'g-', linewidth=2, label='Mean Fitness')
    
    # Plot standard deviation as shaded area
    mean_array = np.array(mean_fitness)
    std_array = np.array(std_fitness)
    ax.fill_between(generations, 
                    mean_array - std_array, 
                    mean_array + std_array,
                    alpha=0.2, color='green', label='±1 Std Dev')
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness (Accuracy)', fontsize=12)
    ax.set_title('Fitness Evolution Over Generations', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Fitness evolution plot saved to {save_path}")
    
    return fig
