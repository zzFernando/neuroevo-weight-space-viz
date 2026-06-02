"""NeuroEvo-Viz — interactive Streamlit tool.

Parameter-driven interface over the paper's two visual components:
  • Aligned UMAP projections (colored by generation / fitness)
  • Population velocity fields (streamlines / quiver)
with a toggle between static (Matplotlib) and interactive (Plotly) rendering.

Run:  pixi run streamlit run app.py     (or: pixi run app)
"""
from __future__ import annotations

import streamlit as st

from benchmarks import DEFAULTS
from utils import run_evolution_benchmark
from visualizations import aligned_umap, vector_field

st.set_page_config(page_title="NeuroEvo-Viz", layout="wide")

PCA_DIMS = {"cifar10": 50}

st.title("NeuroEvo-Viz — Population Dynamics in Neuroevolution")
st.caption("Aligned UMAP embeddings + population velocity fields. "
           "Configure a run in the sidebar and explore the geometry that scalar fitness curves hide.")

# ── Sidebar: run parameters ──────────────────────────────────────────────────
sb = st.sidebar
sb.header("Run parameters")
benchmark = sb.selectbox("Benchmark", list(DEFAULTS.keys()), index=0)
cfg = DEFAULTS[benchmark]
pop_size = sb.slider("Population size", 10, 100, int(cfg["pop_size"]), step=5)
n_generations = sb.slider("Generations", 10, 150, int(cfg["n_generations"]), step=5)
hidden_dim = sb.slider("Hidden dim", 4, 128, int(cfg["hidden_dim"]), step=4)
mutation_rate = sb.slider("Mutation σ", 0.01, 0.30, float(cfg["mutation_rate"]), step=0.01)
seed = sb.number_input("Seed", 0, 9999, 42, step=1)

sb.header("Visualization")
lambda_align = sb.slider("Alignment λ", 0.0, 1.0, 0.8, step=0.1,
                         help="0 = raw UMAP; 1 = collapse to reference (gen 0). Paper uses 0.8.")
render = sb.radio("Render mode", ["Interactive (Plotly)", "Static (Matplotlib)"], index=0)
grid_res = sb.slider("Velocity grid resolution", 8, 40, 22, step=2)
vector_mode = sb.selectbox("Vector style", ["stream", "quiver"], index=0)
interactive = render.startswith("Interactive")

if benchmark in PCA_DIMS:
    sb.info(f"{benchmark}: PCA pre-reduction to {PCA_DIMS[benchmark]} dims; first run downloads data.")


@st.cache_data(show_spinner=True)
def evolve(benchmark, pop_size, n_generations, hidden_dim, mutation_rate, seed):
    res = run_evolution_benchmark(
        benchmark_name=benchmark, pop_size=pop_size, n_generations=n_generations,
        hidden_dim=hidden_dim, mutation_rate=mutation_rate, seed=seed)
    return res.weights_by_gen, res.fitness_by_gen, res.mean_fitness


with st.spinner("Running neuroevolution…"):
    weights_by_gen, fitness_by_gen, mean_fitness = evolve(
        benchmark, pop_size, n_generations, hidden_dim, mutation_rate, int(seed))

pca_dims = PCA_DIMS.get(benchmark)
common = dict(lambda_align=lambda_align, random_state=int(seed), pca_dims=pca_dims)

c1, c2, c3 = st.columns(3)
c1.metric("Final mean fitness", f"{float(mean_fitness[-1]):.3f}")
c2.metric("Generations", len(weights_by_gen))
c3.metric("Weight dim", int(weights_by_gen[0].shape[1]))

tab_vf, tab_umap, tab_fit = st.tabs(["Velocity field", "Aligned UMAP", "Fitness curve"])

with tab_vf:
    st.caption("Mean per-cell displacement between generations — reveals convergence basins and multimodality.")
    if interactive:
        fig = vector_field.plot_interactive(weights_by_gen, fitness_by_gen, grid_res=grid_res,
                                            vector_mode=vector_mode, **common)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = vector_field.plot(weights_by_gen, fitness_by_gen, grid_res=grid_res,
                                vector_mode=vector_mode, **common)
        st.pyplot(fig)

with tab_umap:
    st.caption("Per-generation aligned UMAP embedding, colored by generation (left) and fitness (right).")
    if interactive:
        fig = aligned_umap.plot_interactive(weights_by_gen, fitness_by_gen, **common)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = aligned_umap.plot(weights_by_gen, fitness_by_gen, **common)
        st.pyplot(fig)

with tab_fit:
    st.caption("The scalar view the geometry complements.")
    st.line_chart({"mean fitness": list(map(float, mean_fitness))})
