# Neuroevolution Weight-Space Visualizer

Streamlit app with three visualizations - Aligned UMAP, Vector Field, and Trajectory Bundling - to inspect how a neuroevolved population moves through weight space.

## What's inside
- **Aligned UMAP:** per-generation UMAPs with temporal post-alignment `proj_aligned = proj_k - lambda * (proj_k - proj_ref)`, shown side by side with generation and fitness coloring.
- **Vector Field:** average displacements between consecutive generations aggregated on a 2D grid, rendered as streamlines or quivers with optional smoothing and normalization.
- **Trajectory Bundling:** resampled trajectories, iterative attraction (`beta`) between nearby control points, optional Catmull-Rom smoothing, clustering, and best-trajectory highlighting.
- **Data generation:** lightweight neuroevolution of a one-hidden-layer MLP on `make_moons`; fitness is negative cross-entropy, mutation-only updates.
- **Interactive mode:** toggle in the sidebar to switch between static Matplotlib renders and Plotly charts with hover + zoom.

Project layout:
- `app.py` - Streamlit UI and orchestration.
- `utils.py` - neuroevolution loop plus aligned UMAP helper.
- `visualizations/` - `aligned_umap.py`, `vector_field.py`, `trajectory_bundling.py`, and common plotting helpers.
- `pixi.toml` - environment and task definitions (managed by [pixi](https://pixi.sh)).
- Reference paper (PDF in repo) for the original methodology.

## Install
Requires [pixi](https://pixi.sh/latest/#installation). Tested with Python 3.11.

```bash
pixi install
```

## Run
```bash
pixi run app
```
Or equivalently:
```bash
pixi run streamlit run app.py
```
Click **Run / Refresh** in the sidebar after adjusting parameters; evolution and alignment are cached for faster iteration.

## Available tasks
| Task | Command | Description |
|------|---------|-------------|
| `app` | `pixi run app` | Launch the Streamlit app |
| `figs` | `pixi run figs` | Generate static figures |
| `alignment-demo` | `pixi run alignment-demo` | Run alignment demo script |

## Controls at a glance
- **Neuroevolution:** population size, generations, hidden neurons, mutation rate, seed.
- **UMAP / coloring:** lambda (alignment strength), fitness/generation colormaps, normalization (`power`, `clipped`, `equalized`), gamma, fitness quantile bins.
- **Vector Field:** grid resolution, show points, normalize vectors, Gaussian smoothing (sigma), subsample, mode (`stream` or `quiver`), speed quantization bins.
- **Trajectory Bundling:** control points (K), beta, iterations, neighbor radius (as a fraction of embedding diameter), highlight best trajectory, temporal smoothing window, curve type (`catmull-rom` or `polyline`), clusters, max trajectories rendered.
- **Bilingual UI:** toggle Português/English in the sidebar; each tab describes the visualization and its parameters.

## Notes
- Neighbor radius from the UI is scaled by the embedding diameter before bundling.
- Dependencies are pinned to keep numpy < 2.0 and a matching numba/llvmlite pair; pixi handles isolation automatically so no venv management is needed.
- Numba threading is forced to `omp` to avoid the non-threadsafe `workqueue` backend that can crash under Streamlit reruns. Remove any `NUMBA_NUM_THREADS` env var if set.
- Plotly é usado nas versões interativas; Matplotlib permanece como opção estática.
