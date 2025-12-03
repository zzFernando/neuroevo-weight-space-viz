# Neuroevolution Weight-Space Visualizer

Streamlit app implementing the three visualizations from **Cantareira, Etemad & Paulovich (2020)** — Aligned UMAP, Vector Field, and Trajectory Bundling — to explore how populations move through weight space during neuroevolution.

## Features
- **Aligned UMAP:** per-generation UMAP with temporal post-alignment (`proj_k - λ * (proj_k - proj_ref)`) and dual panels (generation colors, fitness colors).
- **Vector Field:** velocities `α_i(t+1) - α_i(t)` aggregated on a 2D grid and rendered with `streamplot` to show representation flow.
- **Trajectory Bundling:** re-sampled trajectories, iterative attraction (`β`) between nearby control points, and bundled curves colored by mean fitness.

## Project Layout
- `app.py` — Streamlit UI with sidebar controls for mode, λ, grid resolution, β, iterations, and neighbor radius.
- `utils.py` — neuroevolution loop for a make_moons MLP, data containers, aligned UMAP helper.
- `visualizations/` — plotting modules: `aligned_umap.py`, `vector_field.py`, `trajectory_bundling.py`, `__init__.py`.
- `requirements.txt` — Python deps (`streamlit`, `umap-learn`, `numpy`, `matplotlib`, `scikit-learn`, `plotly`, `numba`, `llvmlite`).
- `Cantareira_Etemad_Paulovich___2020___Exploring_neural_network_hidden_layer_activity_using_vector_fields.pdf` — reference paper.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```
Use the sidebar to choose the visualization mode and tweak λ, grid resolution, bundling strength (β), iterations, and neighbor radius fraction.

## Notes
- The app caches evolution runs and alignment for quick iteration.
- Neighbor radius in the UI is interpreted as a fraction of the embedding diameter before being passed to bundling.
- All code is ASCII-only and organized for quick experimentation with Cantareira-style visuals.
