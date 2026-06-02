# Beyond Fitness Curves: Visualizing Population Dynamics in Neuroevolutionary Search

Reproducibility repository for the NeurIPS 2026 submission.

Anonymous code release: https://anonymous.4open.science/r/neurips-2026-9729/

---

## Repository structure

Two top-level buckets:

```
.
├── Docs/                     # everything written
│   ├── papers/               # paper.tex, paper.pdf, .sty, references.bib, checklist.tex, figures/
│   ├── seminars/             # talk materials
│   ├── relatorios/           # experiment_report.html (living narrative)
│   ├── notas/                # supplementary + brax notes
│   ├── referencias/ · dissertacoes/
│   └── EXPERIMENTS.md         # extended-investigation guide (exp 1–30)
└── Experiments/              # everything code + data
    ├── src/                  # shared library (installed editable: benchmarks, visualizations,
    │                         #   utils, shared, paths, neuroevolve_brax)
    ├── apps/                 # Streamlit app + Datashader/Panel explorers
    ├── scripts/              # experiment + ES-generator scripts (01–30, *_evosax, es_sanity, run_all)
    ├── runs/                 # neuroevolution runs + pre-generated datasets (npz)
    ├── results/              # csv metrics (+ results/cache/: cached embeddings)
    ├── figures/              # generated figures
    └── logs/
```

## Environments

Two [pixi](https://pixi.sh) environments in one project (incompatible numpy majors):

| env | flag | purpose | key deps |
|-----|------|---------|----------|
| `viz` (default) | `-e viz` | UMAP / sklearn / datashader analysis, figures, Streamlit | numpy 1.26, umap-learn |
| `evo` | `-e evo` | JAX + brax + evosax run generators | numpy 2.x, jax 0.10, brax 0.14, evosax 0.2 |

`Experiments/src` is installed as an editable package (`neuroevo`) in both envs, so
scripts import `benchmarks`, `utils`, `shared`, `paths` directly — no `sys.path` hacks.

## Reproducing the experiments

```bash
pixi install                      # solves both environments
pixi run -e viz run-all           # core analysis pipeline
pixi run -e evo es-sanity         # validate ES generators (known optimum = 0)
```

See [`Docs/EXPERIMENTS.md`](Docs/EXPERIMENTS.md) for the full exp 1–30 guide.

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
