"""Unit tests for the streamlined neuroevo package."""
from __future__ import annotations

import os
import tempfile
import unittest
import numpy as np

from neuroevo.datasets import generate_complex_dataset
from neuroevo.mlp import FixedMLP
from neuroevo.genetic_algorithm import GeneticAlgorithm
from neuroevo.visualizations import (
    plot_weight_heatmap,
    plot_best_mds,
    plot_population_weight_stats,
)


class TestDataset(unittest.TestCase):
    def test_complex_dataset_shapes(self):
        X_train, X_test, y_train, y_test = generate_complex_dataset(
            n_samples=1200, n_classes=4, seed=7
        )
        self.assertEqual(X_train.ndim, 2)
        self.assertEqual(X_test.ndim, 2)
        self.assertGreater(X_train.shape[0], 0)
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertGreaterEqual(len(np.unique(y_train)), 2)
        self.assertGreaterEqual(len(np.unique(y_test)), 2)
        self.assertEqual(X_train.dtype, np.float32)
        self.assertEqual(X_test.dtype, np.float32)


class TestMLP(unittest.TestCase):
    def test_set_get_weights_roundtrip(self):
        mlp = FixedMLP([5, 8, 3])
        weights = np.random.randn(mlp.total_weights)
        mlp.set_weights(weights)
        recovered = mlp.get_weights()
        np.testing.assert_allclose(weights, recovered)

    def test_forward_batch_shapes(self):
        mlp = FixedMLP([4, 6, 3])
        mlp.set_weights(np.random.randn(mlp.total_weights))
        X = np.random.randn(10, 4)
        out = mlp.forward(X)
        self.assertEqual(out.shape, (10, 3))


class TestGA(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(60, 4))
        self.y = rng.integers(0, 3, size=60)

    def test_run_records_history(self):
        ga = GeneticAlgorithm([4, 10, 3], population_size=12, random_seed=1)
        history = ga.run(self.X, self.y, generations=5, verbose=False)
        # 0..5 inclusive
        self.assertEqual(len(history.generations), 6)
        self.assertEqual(len(ga.best_individual_history), 6)
        self.assertEqual(len(ga.population_history), 6)
        self.assertEqual(len(history.diversity), 6)

    def test_best_payload_bounds(self):
        ga = GeneticAlgorithm([4, 10, 3], population_size=10, random_seed=2)
        ga.run(self.X, self.y, generations=2, verbose=False)
        payload = ga.get_best_individual()
        self.assertEqual(payload["weights"].shape[0], ga.genome_size)
        self.assertGreaterEqual(payload["fitness"], 0.0)
        self.assertLessEqual(payload["fitness"], 1.0)


class TestVisualizations(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.best_history = [rng.normal(size=20) for _ in range(6)]
        self.pop_history = [rng.normal(size=(8, 20)) for _ in range(6)]
        self.fit_history = [rng.random(size=8) for _ in range(6)]
        self.diversity = list(rng.random(size=6))

    def test_visualization_writes_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "heat.png")
            p2 = os.path.join(tmp, "mds.png")
            p3 = os.path.join(tmp, "stats.png")

            plot_weight_heatmap(self.best_history, p1)
            plot_best_mds(self.best_history, p2)
            plot_population_weight_stats(self.best_history, self.pop_history, p3, top_variables=2)

            for path in (p1, p2, p3):
                self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
