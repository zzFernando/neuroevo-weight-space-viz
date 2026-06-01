# Brax HalfCheetah Neuroevolution — UMAP Analysis Pipeline

Drop-in pipeline to add a **third benchmark** to the "Beyond Fitness Curves"
paper, sitting between Make Moons ($d_w=48$) and CIFAR-10 ($d_w \approx 197\text{k}$).
HalfCheetah at $d_w = 390$ (with a 16-unit hidden layer) gives:

- A reinforcement-learning benchmark where neuroevolution is genuinely
  competitive (not a curiosity, like mutation-only on CIFAR-10).
- A fitness curve with **interpretable phases** (random flailing → standing →
  walking → running) — the kind of dynamic regimes the velocity-field
  visualization should capture.
- ~30 minutes wall-clock per seed on Apple M-series CPU (no GPU required).
- Direct alignment with the paper's motivation quoting Lehman et al. and
  Kumar et al. on locomotion / ES.

---

## Files

```
neuroevolve_brax.py   # runs the evolution, saves populations per generation
analyze_umap.py       # single-seed UMAP + velocity-field figure
figure_summary.py     # multi-seed composite (fitness curves + side-by-side VFs)
```

## Install

```bash
pip install "jax[cpu]" brax evosax umap-learn scipy matplotlib
```

(`evosax` is imported as a convenience but the current pipeline uses a plain
NumPy GA matching the paper's setup. Swap in `evosax.algorithms.SimpleGA` or
`Open_ES` later if you want CMA-ES / OpenAI ES baselines.)

## Run

Single seed, conservative ~3-minute setting:

```bash
python neuroevolve_brax.py \
    --pop 40 --gens 50 --episode_len 200 --hidden 16 \
    --seed 42 --out runs/halfcheetah_seed42.npz

python analyze_umap.py runs/halfcheetah_seed42.npz \
    --out figures/halfcheetah_seed42.png
```

Multi-seed (matches Exp 1 of the paper):

```bash
for s in 42 123 7 31 99; do
  python neuroevolve_brax.py --pop 40 --gens 50 --episode_len 200 --hidden 16 \
      --seed $s --out runs/halfcheetah_seed${s}.npz
done
python figure_summary.py --runs_glob 'runs/halfcheetah_seed*.npz' \
    --out figures/halfcheetah_summary.png
```

## Paper-recommended config

For the "production" runs going into the paper, scale up after validating:

| Knob          | Value | Why                                             |
|---------------|-------|-------------------------------------------------|
| `--pop`       | 50    | matches Make Moons, allows fair comparison      |
| `--gens`      | 80    | matches Make Moons, gives full saturation       |
| `--episode_len` | 300 | reduces noise in fitness signal                 |
| `--hidden`    | 16    | keeps $d_w = 390$ — between Make Moons and CIFAR|
| `--sigma`     | 0.05  | matches CIFAR-10 setting; tune if fitness flat  |
| seeds         | 42,123,7,31,99 | same five as Exps 1, for direct comparison |

Estimated total time for the 5-seed sweep on M-series: 30–60 minutes.

## What goes into the paper

1. **Table 1 (multi-seed)** gets a third row:
   `HalfCheetah  |  d_w = 390  |  pop 50  |  gens 80  |  σ 0.05`
   Report spread σ_1, final fitness, convergence gen, same as before.

2. **Figure of velocity fields per seed** (5 columns) — analogous to Fig. 1
   of the paper, showing reproducibility of the flow pattern.

3. **Discussion point**: HalfCheetah sits between Make Moons (compact
   single-basin) and CIFAR-10 (multi-attractor diffuse). Predicted
   behaviour: directional convergence flow with intermediate spread,
   strong fitness localization. If observed, this strengthens claim (iii)
   by giving three points on the dimensional axis instead of two.

4. **Honest framing for limitations**: this still confounds dimensionality
   with task type and mutation rate. A controlled σ sweep on HalfCheetah
   alone would disentangle the σ/dim confound noted in §6 — that becomes
   a natural follow-up experiment.

## Why this beats CIFAR-10 (or can complement it)

- **No PCA pre-reduction.** $d_w = 390$ goes straight into UMAP; no 197k→50
  PCA step that a reviewer can attack as "you're visualizing a projection
  of a projection".
- **Fitness signal is meaningful.** CIFAR-10 final fitness of −2.49 sits
  close to log(10) ≈ −2.30; mutation-only ES barely beats random on
  CIFAR-10. HalfCheetah at fitness ~400 is clearly competent locomotion.
- **Phase structure.** Brax HalfCheetah has well-known learning phases
  that should show up in the velocity field as flow-regime transitions —
  a richer visual phenomenon than CIFAR-10's "rapid collapse to mediocre
  basins".

## Datashader visualization

High-quality static PNG figures using [Datashader](https://datashader.org) — renders
millions of points without overplotting, ideal for large-scale neuroevolution runs.

### Install

Datashader and colorcet are declared in `supplementary/pixi.toml` and installed
automatically with `pixi install`. No extra steps needed when using the pixi environment.

### Usage

**Single seed — 4 PNGs (density, by gen, by fitness, velocity field):**

```bash
cd supplementary
pixi run python ../brax/visualize_datashader.py ../brax/runs/stress_seed42.npz \
    --out_dir ../brax/figures/datashader/
```

Outputs:
- `figures/datashader/stress_seed42_density.png`
- `figures/datashader/stress_seed42_by_gen.png`
- `figures/datashader/stress_seed42_by_fitness.png`
- `figures/datashader/stress_seed42_velocity_field.png`

**Multi-seed composite (1 figure, N rows × 4 cols):**

```bash
pixi run python ../brax/visualize_datashader.py ../brax/runs/stress_seed*.npz \
    --out_dir ../brax/figures/datashader/ --multi_seed_grid
```

Output: `figures/datashader/multi_seed_summary.png`

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--lambda_align` | `0.8` | Alignment strength (Eq. 1 of paper) |
| `--width` / `--height` | `1600` | Single-seed PNG resolution |
| `--subsample N` | off | Fit UMAP on N points, transform the rest |
| `--force_embed` | off | Recompute UMAP even if cache exists |

**Embedding cache:** After the first run, UMAP results are cached to
`runs/<stem>_embedding.npz`. Subsequent runs (e.g., changing `--lambda_align`)
skip UMAP and load the cache instantly.

---

## Stack reference

- `brax 0.14.2` — JAX-based physics, runs natively on CPU/GPU/TPU.
  Note: Brax core marks as "not actively maintained"; the actively
  developed successor is MJX (`mujoco_playground`). The pipeline here
  uses Brax's `generalized` backend which is stable and fine for this
  scale. Migration to MJX is a few-line change if you want to be
  future-proof.
- `evosax` — JAX-based ES library (CMA-ES, OpenAI ES, GA variants).
  Not strictly required for the paper's mutation-only GA but useful
  for ablations.
- `umap-learn 0.5.12` — same as the paper.
