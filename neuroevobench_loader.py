# ============================================================================
# FILENAME: neuroevobench_loader.py
# ============================================================================

"""
NeuroEvoBench data loader for neuroevolution visualization pipeline.

This module provides a robust loader for NeuroEvoBench HDF5 datasets, converting
them to the format expected by the existing visualization pipeline. It handles
data loading, metadata extraction, and preparation for aligned UMAP embeddings
and vector field visualizations.

Dependencies:
- h5py: for HDF5 file reading
- numpy: for array operations
- pandas: for metadata DataFrame
- tqdm: for progress bars
- pathlib: for path handling
- logging: for configurable logging
- umap-learn: for embedding computation (imported in utils)
- jax: optional, for GPU acceleration on M4

Example usage:
    loader = NeuroEvoBenchLoader('./data')
    available = loader.list_available()
    data = loader.load_run('brax_ant_openes_seed0.h5')
    viz_data = loader.prepare_for_visualization(data)
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional JAX import for M4 optimization
try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

from utils import compute_aligned_umap_embedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuroEvoBenchLoader:
    """
    Loader for NeuroEvoBench HDF5 datasets.

    Handles loading, validation, and conversion of neuroevolution data
    to formats compatible with the visualization pipeline.
    """

    def __init__(self, data_dir: str, use_jax: bool = True):
        """
        Initialize the loader with data directory path.

        Args:
            data_dir: Path to directory containing HDF5 files
            use_jax: Whether to use JAX for array operations (M4 optimization)
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")

        self.use_jax = use_jax and JAX_AVAILABLE
        if self.use_jax:
            logger.info("Using JAX for array operations")
        else:
            logger.info("Using NumPy for array operations")

        # Cache for loaded metadata
        self._metadata_cache: Optional[pd.DataFrame] = None

    def _get_array_lib(self):
        """Get the array library to use (jax.numpy or numpy)."""
        return jnp if self.use_jax else np

    def list_available(self) -> pd.DataFrame:
        """
        List all available NeuroEvoBench runs in the data directory.

        Returns:
            DataFrame with columns: filename, task, algorithm, seed,
            n_generations, population_size, n_parameters

        Raises:
            FileNotFoundError: If no HDF5 files found
        """
        if self._metadata_cache is not None:
            return self._metadata_cache

        h5_files = list(self.data_dir.glob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No HDF5 files found in {self.data_dir}")

        metadata = []
        for h5_file in tqdm(h5_files, desc="Scanning HDF5 files"):
            try:
                with h5py.File(h5_file, 'r') as f:
                    attrs = dict(f.attrs)
                    weights_shape = f['weights'].shape
                    fitness_shape = f['fitness'].shape

                    # Validate shapes
                    if len(weights_shape) != 3 or len(fitness_shape) != 2:
                        logger.warning(f"Invalid shape in {h5_file}: weights {weights_shape}, fitness {fitness_shape}")
                        continue

                    n_gens, pop_size, n_params = weights_shape
                    if fitness_shape != (n_gens, pop_size):
                        logger.warning(f"Shape mismatch in {h5_file}: fitness {fitness_shape} vs expected {(n_gens, pop_size)}")
                        continue

                    metadata.append({
                        'filename': h5_file.name,
                        'task': attrs.get('task', 'unknown'),
                        'algorithm': attrs.get('algorithm', 'unknown'),
                        'seed': attrs.get('seed', -1),
                        'n_generations': n_gens,
                        'population_size': pop_size,
                        'n_parameters': n_params,
                        'config': attrs
                    })

            except Exception as e:
                logger.warning(f"Error reading {h5_file}: {e}")
                continue

        if not metadata:
            raise ValueError("No valid HDF5 files found")

        df = pd.DataFrame(metadata)
        self._metadata_cache = df
        return df

    def load_run(self, filename: str, lazy: bool = False) -> Dict:
        """
        Load a specific NeuroEvoBench run from HDF5 file.

        Args:
            filename: Name of the HDF5 file (e.g., 'brax_ant_openes_seed0.h5')
            lazy: If True, load data lazily (for large files)

        Returns:
            Dict with keys: weights_3d, fitness_2d, config, metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            available = self.list_available()['filename'].tolist()
            raise FileNotFoundError(
                f"File {filename} not found in {self.data_dir}. "
                f"Available files: {available[:5]}..."
            )

        try:
            with h5py.File(filepath, 'r') as f:
                logger.info(f"Loading {filename}")

                # Load arrays
                weights = f['weights'][:]
                fitness = f['fitness'][:]

                # Convert to JAX/NumPy
                np_lib = self._get_array_lib()
                weights_3d = np_lib.array(weights)
                fitness_2d = np_lib.array(fitness)

                # Extract metadata
                config = dict(f.attrs)
                n_gens, pop_size, n_params = weights.shape

                metadata = {
                    'filename': filename,
                    'n_generations': n_gens,
                    'population_size': pop_size,
                    'n_parameters': n_params,
                    'task': config.get('task', 'unknown'),
                    'algorithm': config.get('algorithm', 'unknown'),
                    'seed': config.get('seed', -1),
                    'file_size_mb': filepath.stat().st_size / (1024 * 1024)
                }

                return {
                    'weights_3d': weights_3d,
                    'fitness_2d': fitness_2d,
                    'config': config,
                    'metadata': metadata
                }

        except Exception as e:
            raise ValueError(f"Error loading {filename}: {e}")

    def load_multiple_seeds(self, task: str, algorithm: str,
                           seeds: List[int]) -> List[Dict]:
        """
        Load multiple seeds for the same task and algorithm.

        Args:
            task: Task name (e.g., 'brax_ant')
            algorithm: Algorithm name (e.g., 'openes')
            seeds: List of seed numbers

        Returns:
            List of data dicts, one per seed
        """
        available = self.list_available()
        data_list = []

        for seed in seeds:
            # Find matching file
            mask = (
                (available['task'] == task) &
                (available['algorithm'] == algorithm) &
                (available['seed'] == seed)
            )
            matches = available[mask]

            if len(matches) == 0:
                logger.warning(f"No file found for {task}_{algorithm}_seed{seed}")
                continue
            elif len(matches) > 1:
                logger.warning(f"Multiple files found for {task}_{algorithm}_seed{seed}, using first")
                filename = matches.iloc[0]['filename']
            else:
                filename = matches.iloc[0]['filename']

            try:
                data = self.load_run(filename)
                data_list.append(data)
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")

        return data_list

    def prepare_for_visualization(self, data: Dict, cache_dir: Optional[str] = None,
                                lambda_align: float = 0.3, random_state: int = 42) -> Dict:
        """
        Prepare loaded data for visualization pipeline.

        Computes aligned UMAP embeddings and formats data as expected by
        visualization functions.

        Args:
            data: Output from load_run()
            cache_dir: Directory to cache computed embeddings (optional)
            lambda_align: Alignment parameter for UMAP
            random_state: Random seed for UMAP

        Returns:
            Dict in VIZ_DATA format with keys: weights_3d, fitness_2d,
            embeddings_2d, generation_ids, individual_ids, config,
            n_generations, population_size, n_parameters
        """
        weights_3d = data['weights_3d']
        fitness_2d = data['fitness_2d']
        metadata = data['metadata']

        n_gens, pop_size, n_params = weights_3d.shape

        # Convert to list of arrays for UMAP computation
        weights_by_gen = [weights_3d[g] for g in range(n_gens)]
        fitness_by_gen = [fitness_2d[g] for g in range(n_gens)]
        cache_path = None
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(exist_ok=True)
            cache_name = f"{metadata['filename']}_embeddings_lambda{lambda_align}_seed{random_state}.npy"
            cache_path = cache_dir / cache_name

            if cache_path.exists():
                logger.info(f"Loading cached embeddings from {cache_path}")
                embeddings_2d = np.load(cache_path)
            else:
                embeddings_2d = None
        else:
            embeddings_2d = None

        if embeddings_2d is None:
            logger.info("Computing aligned UMAP embeddings")
            try:
                embedding_all, gen_labels, per_gen_embeddings = compute_aligned_umap_embedding(
                    weights_by_gen, lambda_align=lambda_align, random_state=random_state
                )

                # Reshape to (n_gens, pop_size, 2)
                embeddings_2d = np.array(per_gen_embeddings).reshape(n_gens, pop_size, 2)

                # Cache if requested
                if cache_path:
                    np.save(cache_path, embeddings_2d)
                    logger.info(f"Cached embeddings to {cache_path}")

            except Exception as e:
                logger.error(f"UMAP computation failed: {e}")
                # Fallback: use random 2D projections
                logger.warning("Using random 2D projections as fallback")
                embeddings_2d = np.random.randn(n_gens, pop_size, 2)

        # Create generation and individual IDs
        generation_ids = np.repeat(np.arange(n_gens), pop_size)
        individual_ids = np.tile(np.arange(pop_size), n_gens)

        return {
            'weights_3d': weights_3d,
            'fitness_2d': fitness_2d,
            'embeddings_2d': embeddings_2d,
            'weights_by_gen': weights_by_gen,
            'fitness_by_gen': fitness_by_gen,
            'generation_ids': generation_ids,
            'individual_ids': individual_ids,
            'config': data['config'],
            'n_generations': n_gens,
            'population_size': pop_size,
            'n_parameters': n_params
        }
