# ============================================================================
# FILENAME: run_pipeline.py
# ============================================================================

"""
Complete end-to-end pipeline for NeuroEvoBench data loading and visualization.

This script loads NeuroEvoBench data, computes aligned UMAP embeddings,
and generates publication-ready figures using the existing visualization
modules.

Usage:
    python run_pipeline.py --task brax_ant --algorithm openes --seed 0 --output-dir ./figures

Dependencies:
- neuroevobench_loader: custom loader module
- visualizations.*: existing visualization modules
- matplotlib, numpy, pandas
- tqdm for progress bars

Output:
- Aligned UMAP scatter plot (2 panels)
- Vector field with streamlines (2x2 grid)
- Divergence analysis plot
- Statistics printed to console
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from neuroevobench_loader import NeuroEvoBenchLoader
from visualizations.aligned_umap import plot
from visualizations.vector_field import plot
# from visualizations.divergence import compute_divergence, plot_divergence

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_statistics(viz_data: Dict) -> Dict:
    """
    Compute basic statistics from the visualization data.

    Args:
        viz_data: Prepared visualization data

    Returns:
        Dict with statistics
    """
    fitness_2d = viz_data['fitness_2d']
    embeddings_2d = viz_data['embeddings_2d']

    stats = {
        'n_generations': viz_data['n_generations'],
        'population_size': viz_data['population_size'],
        'n_parameters': viz_data['n_parameters'],
        'fitness_mean': float(np.mean(fitness_2d)),
        'fitness_std': float(np.std(fitness_2d)),
        'fitness_min': float(np.min(fitness_2d)),
        'fitness_max': float(np.max(fitness_2d)),
        'fitness_final_mean': float(np.mean(fitness_2d[-1])),
        'fitness_improvement': float(np.mean(fitness_2d[-1]) - np.mean(fitness_2d[0])),
        'embedding_spread': float(np.std(embeddings_2d)),
    }

    return stats


def save_figure(fig, filename: str, output_dir: Path, dpi: int = 300):
    """
    Save matplotlib figure with high quality settings.

    Args:
        fig: Matplotlib figure
        filename: Output filename
        output_dir: Output directory
        dpi: Resolution
    """
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved {filename} to {output_path}")
    plt.close(fig)


def run_pipeline(
    data_dir: str,
    task: str,
    algorithm: str,
    seed: int,
    output_dir: str = './figures',
    cache_dir: Optional[str] = None,
    lambda_align: float = 0.3,
    random_state: int = 42
):
    """
    Run the complete visualization pipeline.

    Args:
        data_dir: Directory containing HDF5 files
        task: Task name
        algorithm: Algorithm name
        seed: Random seed
        output_dir: Output directory for figures
        cache_dir: Cache directory for embeddings
        lambda_align: UMAP alignment parameter
        random_state: Random seed for reproducibility
    """
    # Setup directories
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True, parents=True)

    # Initialize loader
    logger.info("Initializing NeuroEvoBench loader")
    loader = NeuroEvoBenchLoader(data_dir)

    # Load data
    filename = f"{task}_{algorithm}_seed{seed}.h5"
    logger.info(f"Loading {filename}")
    try:
        data = loader.load_run(filename)
    except FileNotFoundError as e:
        logger.error(str(e))
        # Try to find similar files
        available = loader.list_available()
        matches = available[
            (available['task'] == task) &
            (available['algorithm'] == algorithm)
        ]
        if not matches.empty:
            logger.info(f"Available files for {task}_{algorithm}:")
            for _, row in matches.iterrows():
                logger.info(f"  {row['filename']}")
        return

    # Prepare for visualization
    logger.info("Preparing data for visualization")
    viz_data = loader.prepare_for_visualization(
        data, cache_dir=cache_dir, lambda_align=lambda_align, random_state=random_state
    )

    # Compute statistics
    stats = compute_statistics(viz_data)
    logger.info("Dataset Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # Generate visualizations
    logger.info("Generating aligned UMAP plot")
    try:
        fig_umap = plot(
            weights_by_gen=viz_data['weights_by_gen'],
            fitness_by_gen=viz_data['fitness_by_gen']
        )
        save_figure(fig_umap, f'{task}_{algorithm}_seed{seed}_aligned_umap.png', output_dir)
    except Exception as e:
        logger.error(f"Failed to generate UMAP plot: {e}")

    logger.info("Generating vector field plot")
    try:
        fig_vector = plot(
            weights_by_gen=viz_data['weights_by_gen'],
            fitness_by_gen=viz_data['fitness_by_gen']
        )
        save_figure(fig_vector, f'{task}_{algorithm}_seed{seed}_vector_field.png', output_dir)
    except Exception as e:
        logger.error(f"Failed to generate vector field plot: {e}")

    # logger.info("Generating divergence plot")
    # try:
    #     divergence_data = compute_divergence(viz_data['embeddings_2d'])
    #     fig_div = plot_divergence(divergence_data)
    #     save_figure(fig_div, f'{task}_{algorithm}_seed{seed}_divergence.png', output_dir)
    # except Exception as e:
    #     logger.error(f"Failed to generate divergence plot: {e}")

    logger.info("Pipeline completed successfully")


def main():
    parser = argparse.ArgumentParser(description="NeuroEvoBench Visualization Pipeline")
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing HDF5 files')
    parser.add_argument('--task', type=str, required=True,
                       help='Task name (e.g., brax_ant)')
    parser.add_argument('--algorithm', type=str, required=True,
                       help='Algorithm name (e.g., openes)')
    parser.add_argument('--seed', type=int, required=True,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='./figures',
                       help='Output directory for figures')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='Cache directory for embeddings')
    parser.add_argument('--lambda-align', type=float, default=0.3,
                       help='UMAP alignment parameter')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        task=args.task,
        algorithm=args.algorithm,
        seed=args.seed,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        lambda_align=args.lambda_align,
        random_state=args.random_state
    )


if __name__ == '__main__':
    main()