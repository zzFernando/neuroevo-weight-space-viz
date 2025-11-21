"""Genetic Algorithm for evolving neural network weights."""
import numpy as np
from .mlp import FixedMLP


class GeneticAlgorithm:
    """Simple genetic algorithm for evolving MLP weights."""
    
    def __init__(self, 
                 layer_sizes,
                 population_size=50,
                 mutation_rate=0.1,
                 mutation_std=0.5,
                 elite_size=5,
                 tournament_size=3,
                 random_seed=None):
        """
        Initialize the genetic algorithm.
        
        Args:
            layer_sizes: Architecture of the MLP (e.g., [4, 10, 3])
            population_size: Number of individuals in the population
            mutation_rate: Probability of mutating each weight
            mutation_std: Standard deviation of mutation noise
            elite_size: Number of top individuals to preserve
            tournament_size: Size of tournament for selection
            random_seed: Random seed for reproducibility
        """
        self.layer_sizes = layer_sizes
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Create template MLP to get weight dimensions
        self.mlp_template = FixedMLP(layer_sizes)
        self.genome_size = self.mlp_template.total_weights
        
        # Initialize population
        self.population = self._initialize_population()
        self.fitness_scores = np.zeros(population_size)
        
        # Tracking for logging
        self.generation = 0
        self.best_individual_history = []
        self.population_history = []
        self.fitness_history = []
    
    def _initialize_population(self):
        """Initialize population with random weights."""
        # Xavier initialization scaled for the genome
        scale = np.sqrt(2.0 / (self.layer_sizes[0] + self.layer_sizes[-1]))
        return np.random.randn(self.population_size, self.genome_size) * scale
    
    def evaluate_fitness(self, X, y):
        """
        Evaluate fitness of all individuals in the population.
        
        Args:
            X: Training data, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)
        """
        mlp = FixedMLP(self.layer_sizes)
        
        for i in range(self.population_size):
            mlp.set_weights(self.population[i])
            predictions = mlp.predict(X)
            accuracy = np.mean(predictions == y)
            self.fitness_scores[i] = accuracy
    
    def _tournament_selection(self):
        """Select an individual using tournament selection."""
        tournament_indices = np.random.choice(
            self.population_size, 
            size=self.tournament_size, 
            replace=False
        )
        tournament_fitness = self.fitness_scores[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx].copy()
    
    def _crossover(self, parent1, parent2):
        """Perform uniform crossover between two parents."""
        mask = np.random.rand(self.genome_size) < 0.5
        child = np.where(mask, parent1, parent2)
        return child
    
    def _mutate(self, individual):
        """Apply Gaussian mutation to an individual."""
        mutation_mask = np.random.rand(self.genome_size) < self.mutation_rate
        mutations = np.random.randn(self.genome_size) * self.mutation_std
        individual[mutation_mask] += mutations[mutation_mask]
        return individual
    
    def evolve_generation(self):
        """Evolve one generation."""
        # Sort population by fitness
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        
        # Create new population
        new_population = []
        
        # Elitism: preserve best individuals
        for i in range(self.elite_size):
            new_population.append(self.population[sorted_indices[i]].copy())
        
        # Generate offspring
        while len(new_population) < self.population_size:
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            new_population.append(child)
        
        self.population = np.array(new_population)
        self.generation += 1
    
    def get_best_individual(self):
        """Get the best individual and its fitness."""
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx].copy(), self.fitness_scores[best_idx]
    
    def log_generation(self):
        """Log current generation's best individual and population."""
        best_individual, best_fitness = self.get_best_individual()
        
        self.best_individual_history.append(best_individual.copy())
        self.population_history.append(self.population.copy())
        self.fitness_history.append(self.fitness_scores.copy())
        
        return {
            'generation': self.generation,
            'best_fitness': best_fitness,
            'mean_fitness': np.mean(self.fitness_scores),
            'std_fitness': np.std(self.fitness_scores)
        }
    
    def run(self, X, y, generations=100, verbose=True):
        """
        Run the genetic algorithm.
        
        Args:
            X: Training data
            y: Training labels
            generations: Number of generations to evolve
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
        """
        history = {
            'generation': [],
            'best_fitness': [],
            'mean_fitness': [],
            'std_fitness': []
        }
        
        # Evaluate initial population
        self.evaluate_fitness(X, y)
        stats = self.log_generation()
        
        if verbose:
            print(f"Gen {stats['generation']:3d}: "
                  f"Best={stats['best_fitness']:.4f}, "
                  f"Mean={stats['mean_fitness']:.4f}, "
                  f"Std={stats['std_fitness']:.4f}")
        
        for key in stats:
            history[key].append(stats[key])
        
        # Evolution loop
        for gen in range(generations):
            self.evolve_generation()
            self.evaluate_fitness(X, y)
            stats = self.log_generation()
            
            if verbose and (gen + 1) % 10 == 0:
                print(f"Gen {stats['generation']:3d}: "
                      f"Best={stats['best_fitness']:.4f}, "
                      f"Mean={stats['mean_fitness']:.4f}, "
                      f"Std={stats['std_fitness']:.4f}")
            
            for key in stats:
                history[key].append(stats[key])
        
        return history
