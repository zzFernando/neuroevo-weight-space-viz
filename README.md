# Neuroevolution Weight Space Visualization

This project focuses on *how* neural weights move and organize during neuroevolution. We evolve a fixed-topology MLP on a deliberately complex dataset and export visualizations that highlight the dynamics of each generation.

## What's Included

- **Complex dataset generator** – mixes spirals, moons, circles and noisy blobs into a high-dimensional classification challenge (`neuroevo/datasets.py`). The problem is hard enough to justify the visualization effort.
- **Minimal GA** – a lean yet fully traceable genetic algorithm that keeps every population, best genome and diversity score (`neuroevo/genetic_algorithm.py`).
- **Visualization suite** – four plots purpose-built for weight-space analysis (`neuroevo/visualizations.py`):
  1. PCA trajectory of the best genome across generations.
  2. Heatmap of the most dynamic weights, generation vs. weight index.
  3. PCA projection of whole populations at different checkpoints.
  4. Dual-axis curve showing best fitness vs. mean pairwise diversity.

These artifacts aim to contribute new ideas to the neuroevolution visualization literature, especially for tracking weight evolution rather than just fitness curves.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py \
  --generations 180 \
  --population-size 70 \
  --samples 6000 \
  --seed 0
```

Outputs are saved under `results/`:

| File | Description |
| --- | --- |
| `weights_heatmap.png` | Heatmap (gerações × variáveis) do melhor indivíduo |
| `weights_mds.png` | Trajetória 2-D via MDS dos vetores de pesos |
| `population_weight_stats.png` | Linhas com média ± desvio da população vs. melhor indivíduo |

## Project Structure

```
neuroevo-weight-space-viz/
├── main.py
├── neuroevo/
│   ├── __init__.py
│   ├── datasets.py         # complex dataset generator
│   ├── genetic_algorithm.py
│   ├── mlp.py
│   └── visualizations.py
├── results/                # generated artifacts
├── tests/
└── README.md
```

## Extending

- Modify `generate_complex_dataset` to experiment with other manifolds.
- Adjust GA hyperparameters via CLI flags (`--mutation-rate`, `--mutation-std`, `--elite-size`, etc.).
- Add new visualization ideas in `neuroevo/visualizations.py`; all histories are already stored.

## Testing

```bash
python -m unittest discover tests -v
```

The suite checks dataset properties, GA bookkeeping and visualization smoke tests.

---

Created for research on novel visualizations for neuroevolution weight dynamics.
