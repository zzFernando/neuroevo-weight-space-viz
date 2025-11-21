# Neuroevolution Weight Space Visualization

Visualization prototypes for neuroevolution algorithms that evolve only neural network weights: trajectory in weight space (MDS) and population diversity (UMAP).

This project implements a simple genetic algorithm for evolving the weights of a fixed-topology Multi-Layer Perceptron (MLP) on a toy classification task. It provides comprehensive visualizations to understand the evolution process in weight space.

## Features

- **Fixed-Topology MLP**: Neural network with configurable layer sizes
- **Genetic Algorithm**: Evolves only the weights (genotype = flattened weights) with:
  - Tournament selection
  - Uniform crossover
  - Gaussian mutation
  - Elitism
- **Fitness Evaluation**: Accuracy on toy classification task
- **Logging**: Tracks best individual and population each generation
- **Visualizations**:
  1. **MDS 2D Trajectory**: Shows the path of the best individual through weight space across generations
  2. **UMAP 2D Projections**: Population snapshots at start/mid/end colored by fitness to study diversity and convergence
  3. **Fitness Evolution**: Best and mean fitness over generations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the main script to execute the genetic algorithm and generate visualizations:

```bash
python main.py
```

This will:
1. Create a toy classification dataset (500 samples, 4 features, 3 classes)
2. Initialize and run the genetic algorithm for 100 generations
3. Generate three visualization plots saved to `results/` directory:
   - `fitness_evolution.png`: Fitness progression over generations
   - `mds_trajectory.png`: MDS 2D trajectory of best individual in weight space
   - `umap_snapshots.png`: UMAP projections of population at start/mid/end

## Project Structure

```
neuroevo-weight-space-viz/
├── neuroevo/
│   ├── __init__.py
│   ├── mlp.py                  # Fixed-topology MLP implementation
│   ├── genetic_algorithm.py    # Genetic algorithm for weight evolution
│   └── visualizations.py       # Visualization functions (MDS, UMAP, fitness)
├── main.py                      # Main script to run experiments
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Algorithm Details

### MLP Architecture
- Configurable layer sizes (default: [4, 10, 3] for input, hidden, output)
- ReLU activation for hidden layers
- No activation on output layer
- Weights and biases as genotype

### Genetic Algorithm Parameters
- Population size: 50
- Mutation rate: 0.1
- Mutation standard deviation: 0.5
- Elite size: 5
- Tournament size: 3

### Visualizations

**MDS Trajectory**: Uses Multidimensional Scaling to project the high-dimensional weight vectors of the best individual from each generation into 2D space, revealing the evolutionary path through weight space.

**UMAP Snapshots**: Uses UMAP (Uniform Manifold Approximation and Projection) to visualize the entire population at three time points:
- Start: Initial random population
- Mid: Population halfway through evolution
- End: Final converged population

Points are colored by fitness (green = high, red = low), showing how the population:
- Initially has high diversity (scattered points)
- Gradually converges toward high-fitness regions
- Eventually clusters around optimal weight configurations

## Example Results

The genetic algorithm typically achieves:
- Train accuracy: ~80-85%
- Test accuracy: ~75-80%

The visualizations show:
- Clear evolutionary trajectory toward better fitness regions
- Gradual population convergence from diverse to clustered
- Correlation between weight space proximity and fitness

## Customization

You can modify the experiment parameters in `main.py`:

```python
# Dataset parameters
n_samples, n_features, n_classes = 500, 4, 3

# MLP architecture
layer_sizes = [n_features, 10, n_classes]

# GA parameters
ga = GeneticAlgorithm(
    layer_sizes=layer_sizes,
    population_size=50,
    mutation_rate=0.1,
    mutation_std=0.5,
    elite_size=5,
    tournament_size=3,
    random_seed=42
)

# Training
n_generations = 100
```

See `example_custom.py` for an example using the moons dataset with different parameters.

## Requirements

- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- umap-learn >= 0.5.5

## Testing

Run the unit tests to verify the implementation:

```bash
python -m unittest discover tests -v
```

All tests should pass, covering:
- MLP forward/backward pass
- Weight initialization and serialization
- Genetic algorithm operations
- Evolution and fitness evaluation
- Edge case handling

## License

See LICENSE file for details.

## Author

Fernando - Master's Thesis Project on Neuroevolution

