# Supplementary Code — Beyond Fitness Curves

Reproducibility package for the paper  
**"Beyond Fitness Curves: Visualizing Population Dynamics in Neuroevolutionary Search"**  
Submitted to NeurIPS 2026.

---

## Requirements

Install [pixi](https://prefix.dev/docs/pixi/overview) (cross-platform conda environment manager):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then install the environment (one command, no manual dependency management):

```bash
pixi install
```

This installs Python 3.11 with all required packages (numpy, scikit-learn, umap-learn, matplotlib, tqdm, scipy) as specified in `pixi.toml`.

---

## Reproducing all experiments

Run all four experiments sequentially (uses cached results if available):

```bash
pixi run python -m experiments.run_all
```

Force a full re-run from scratch:

```bash
pixi run python -m experiments.run_all --force-rerun
```

Results are written to `results/` and figures to `figures/`.

---

## Individual experiments

| Script | Paper section | Output |
|--------|--------------|--------|
| `experiments/01_multi_seed_robustness.py` | §4.1 | `exp1_multiseed.png`, `exp1_multiseed_metrics.csv` |
| `experiments/02_projection_baselines.py` | §4.2 | `exp2_projection_baselines.png`, `exp2_projection_metrics.csv` |
| `experiments/03_alignment_ablation.py` | §4.3 | `exp3_alignment_ablation.png`, `exp3_tc_curve.png`, `exp3_alignment_coherence.csv` |
| `experiments/04_synthetic_unimodal_test.py` | §4.4 | `exp4_synthetic_validation.png`, `exp4_synthetic_validation.csv` |

Run individually:

```bash
pixi run python -m experiments.01_multi_seed_robustness
pixi run python -m experiments.02_projection_baselines
pixi run python -m experiments.03_alignment_ablation
pixi run python -m experiments.04_synthetic_unimodal_test
```

Regenerate the pipeline diagram (Figure 1):

```bash
pixi run python generate_pipeline_diagram.py
```

---

## Compute requirements

All experiments run on a single CPU. No GPU required.

| Experiment | Approx. time |
|------------|-------------|
| Exp 1 (Make Moons, 5 seeds) | ~1 min |
| Exp 1 (CIFAR-10, 5 seeds) | ~20 min |
| Exp 2 (projection baselines) | ~1–2 min |
| Exp 3 (alignment ablation) | ~1–2 min |
| Exp 4 (synthetic validation) | ~1 min |
| **Total** | **≲ 30 min** |

Tested on Apple M-series CPU, 16 GB RAM.

---

## Code structure

```
.
├── pixi.toml                        # reproducible environment
├── utils.py                         # evolution runner + aligned UMAP embedding
├── generate_pipeline_diagram.py     # generates Figure 1
├── benchmarks/
│   ├── base.py                      # NeuroEvoBase class
│   ├── moons.py                     # Make Moons benchmark
│   └── cifar10.py                   # CIFAR-10 benchmark
├── experiments/
│   ├── shared.py                    # shared helpers (metrics, plot functions)
│   ├── run_all.py                   # run all experiments sequentially
│   ├── 01_multi_seed_robustness.py
│   ├── 02_projection_baselines.py
│   ├── 03_alignment_ablation.py
│   └── 04_synthetic_unimodal_test.py
└── visualizations/
    ├── vector_field.py              # velocity grid computation
    └── aligned_umap.py              # aligned UMAP utilities
```

---

## Datasets

All datasets are standard public benchmarks downloaded automatically at runtime:

- **Make Moons** — generated via `sklearn.datasets.make_moons` (no download required)
- **CIFAR-10** — downloaded automatically via `urllib` from the Toronto mirror on first run (~170 MB, cached in `$HOME/.cache/neuroevo_viz/`)
