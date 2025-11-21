"""Unit tests for the neuroevolution package."""
import unittest
import numpy as np
from neuroevo.mlp import FixedMLP
from neuroevo.genetic_algorithm import GeneticAlgorithm


class TestFixedMLP(unittest.TestCase):
    """Test cases for FixedMLP class."""
    
    def test_initialization(self):
        """Test MLP initialization."""
        mlp = FixedMLP([4, 10, 3])
        self.assertEqual(mlp.layer_sizes, [4, 10, 3])
        self.assertEqual(mlp.num_layers, 2)
        # 4*10 + 10 (first layer) + 10*3 + 3 (second layer) = 40 + 10 + 30 + 3 = 83
        self.assertEqual(mlp.total_weights, 83)
    
    def test_set_and_get_weights(self):
        """Test setting and getting weights."""
        mlp = FixedMLP([2, 3, 2])
        # 2*3 + 3 + 3*2 + 2 = 6 + 3 + 6 + 2 = 17
        weights = np.random.randn(17)
        mlp.set_weights(weights)
        retrieved = mlp.get_weights()
        np.testing.assert_array_almost_equal(weights, retrieved)
    
    def test_forward_pass(self):
        """Test forward pass."""
        mlp = FixedMLP([2, 3, 2])
        mlp.set_weights(np.random.randn(17))
        
        # Single sample
        X_single = np.array([1.0, 2.0])
        output_single = mlp.forward(X_single)
        self.assertEqual(output_single.shape, (2,))
        
        # Multiple samples
        X_batch = np.array([[1.0, 2.0], [3.0, 4.0]])
        output_batch = mlp.forward(X_batch)
        self.assertEqual(output_batch.shape, (2, 2))
    
    def test_predict(self):
        """Test prediction."""
        mlp = FixedMLP([2, 3, 2])
        mlp.set_weights(np.random.randn(17))
        
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        predictions = mlp.predict(X)
        self.assertEqual(predictions.shape, (2,))
        self.assertTrue(all(p in [0, 1] for p in predictions))
    
    def test_uninitialized_weights_error(self):
        """Test that using MLP without setting weights raises error."""
        mlp = FixedMLP([2, 3, 2])
        
        with self.assertRaises(AttributeError):
            mlp.forward(np.array([1.0, 2.0]))
        
        with self.assertRaises(AttributeError):
            mlp.get_weights()
    
    def test_wrong_weight_size(self):
        """Test that setting wrong number of weights raises error."""
        mlp = FixedMLP([2, 3, 2])
        
        with self.assertRaises(ValueError):
            mlp.set_weights(np.random.randn(10))  # Wrong size


class TestGeneticAlgorithm(unittest.TestCase):
    """Test cases for GeneticAlgorithm class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create simple dataset
        np.random.seed(42)
        self.X = np.random.randn(100, 4)
        self.y = np.random.randint(0, 3, 100)
    
    def test_initialization(self):
        """Test GA initialization."""
        ga = GeneticAlgorithm([4, 10, 3], population_size=20, random_seed=42)
        self.assertEqual(ga.population_size, 20)
        self.assertEqual(ga.population.shape, (20, 83))
        self.assertEqual(len(ga.fitness_scores), 20)
    
    def test_fitness_evaluation(self):
        """Test fitness evaluation."""
        ga = GeneticAlgorithm([4, 10, 3], population_size=10, random_seed=42)
        ga.evaluate_fitness(self.X, self.y)
        
        # All fitness scores should be between 0 and 1
        self.assertTrue(all(0 <= f <= 1 for f in ga.fitness_scores))
    
    def test_evolution(self):
        """Test that evolution improves fitness."""
        ga = GeneticAlgorithm([4, 10, 3], population_size=20, random_seed=42)
        
        # Initial fitness
        ga.evaluate_fitness(self.X, self.y)
        initial_best = np.max(ga.fitness_scores)
        
        # Evolve for a few generations
        for _ in range(10):
            ga.evolve_generation()
            ga.evaluate_fitness(self.X, self.y)
        
        final_best = np.max(ga.fitness_scores)
        
        # Fitness should generally improve (or at least not get worse with elitism)
        self.assertGreaterEqual(final_best, initial_best)
    
    def test_best_individual(self):
        """Test getting best individual."""
        ga = GeneticAlgorithm([4, 10, 3], population_size=10, random_seed=42)
        ga.evaluate_fitness(self.X, self.y)
        
        best_ind, best_fit = ga.get_best_individual()
        self.assertEqual(len(best_ind), 83)
        self.assertEqual(best_fit, np.max(ga.fitness_scores))
    
    def test_logging(self):
        """Test generation logging."""
        ga = GeneticAlgorithm([4, 10, 3], population_size=10, random_seed=42)
        ga.evaluate_fitness(self.X, self.y)
        
        stats = ga.log_generation()
        self.assertIn('generation', stats)
        self.assertIn('best_fitness', stats)
        self.assertIn('mean_fitness', stats)
        self.assertIn('std_fitness', stats)
        
        self.assertEqual(len(ga.best_individual_history), 1)
        self.assertEqual(len(ga.population_history), 1)
        self.assertEqual(len(ga.fitness_history), 1)
    
    def test_run_method(self):
        """Test full run."""
        ga = GeneticAlgorithm([4, 10, 3], population_size=10, random_seed=42)
        history = ga.run(self.X, self.y, generations=5, verbose=False)
        
        # Check history structure
        self.assertEqual(len(history['generation']), 6)  # 0 + 5 generations
        self.assertEqual(len(history['best_fitness']), 6)
        self.assertEqual(len(history['mean_fitness']), 6)
        
        # Check that we logged everything
        self.assertEqual(len(ga.best_individual_history), 6)
        self.assertEqual(len(ga.population_history), 6)
    
    def test_tournament_size_edge_case(self):
        """Test tournament selection with large tournament size."""
        # Tournament size larger than population should be handled safely
        ga = GeneticAlgorithm(
            [4, 10, 3], 
            population_size=5, 
            tournament_size=10,  # Larger than population
            random_seed=42
        )
        ga.evaluate_fitness(self.X, self.y)
        
        # This should not raise an error
        ga.evolve_generation()
        ga.evaluate_fitness(self.X, self.y)
        
        # Should still work
        best_ind, best_fit = ga.get_best_individual()
        self.assertIsNotNone(best_ind)


if __name__ == '__main__':
    unittest.main()
