# Beyond Fitness Curves: Visualizing Population Dynamics in Neuroevolutionary Search

Reproducibility repository for the NeurIPS 2026 submission.

Anonymous code release: https://anonymous.4open.science/r/neurips-2026-9729/

---

## Repository structure

```
.
├── paper.tex                  # main paper
├── paper.pdf                  # compiled PDF
├── checklist.tex              # NeurIPS reproducibility checklist
├── neurips_2026.sty           # NeurIPS 2026 style file
├── figures/                   # all figures used in the paper
└── supplementary/             # self-contained reproducibility package
    ├── README.md              # instructions for running experiments
    ├── pixi.toml              # reproducible environment (pixi)
    ├── utils.py               # evolution runner + aligned UMAP embedding
    ├── generate_pipeline_diagram.py
    ├── benchmarks/            # Make Moons and CIFAR-10 benchmarks
    ├── experiments/           # four experiment scripts + run_all.py
    └── visualizations/        # vector field and aligned UMAP utilities
```

## Reproducing the experiments

See [`supplementary/README.md`](supplementary/README.md) for full instructions.

```bash
cd supplementary
pixi install
pixi run python -m experiments.run_all
```

All experiments run on a single CPU in under 30 minutes.
