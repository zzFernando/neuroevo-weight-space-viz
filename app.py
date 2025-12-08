# app.py
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils import compute_aligned_umap_embedding, run_evolution
from visualizations.aligned_umap import plot as plot_aligned_umap
from visualizations.common import build_fitness_quantile_colormap, get_discrete_cmap
from visualizations.trajectory_bundling import plot as plot_trajectory_bundling
from visualizations.vector_field import plot as plot_vector_field


st.set_page_config(page_title="Neuroevo Weight-Space Viz", layout="wide")

st.title("Exploring Neuroevolution Weight Space (Cantareira 2020)")
st.markdown(
    """
    Faithful implementations of the three visualizations from
    **Cantareira, Etemad & Paulovich (2020)** to explore trajectories in
    weight-space representations: **Aligned UMAP**, **Vector Field**, and
    **Trajectory Bundling**.
    """
)


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header("Neuroevolution Parameters")
pop_size = st.sidebar.slider("Population size", 20, 400, 120, step=20)
n_generations = st.sidebar.slider("Number of generations", 5, 80, 25, step=1)
hidden_dim = st.sidebar.slider("Hidden neurons", 4, 128, 32, step=4)
mutation_rate = st.sidebar.slider("Mutation rate", 0.005, 0.2, 0.05, step=0.005)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42)

st.sidebar.header("Visualizations")
lambda_align = st.sidebar.slider("lambda (UMAP alignment)", 0.0, 1.0, 0.3, step=0.05)
cmap_options = ["plasma", "inferno", "magma", "viridis", "cividis", "turbo", "fitness_map"]
fitness_cmap_name = st.sidebar.selectbox("Fitness colormap", cmap_options, index=1)
gen_cmap_name = st.sidebar.selectbox("Generation colormap", cmap_options, index=0)
norm_mode = st.sidebar.selectbox("Color normalization", ["power", "clipped", "equalized"], index=0)
gamma = st.sidebar.slider("Power gamma (fitness)", 0.1, 1.0, 0.3, step=0.05)
fitness_bins = st.sidebar.slider("Fitness quantile bins", 5, 15, 9, step=1)

st.sidebar.subheader("Vector Field")
grid_res = st.sidebar.slider("Grid resolution (vector field)", 5, 60, 20, step=1)
show_points = st.sidebar.checkbox("Show generation points", value=True)
normalize_vectors = st.sidebar.checkbox("Normalize vectors (unit length)", value=False)
smoothing_sigma = st.sidebar.slider("Vector smoothing (sigma)", 0.0, 3.0, 1.0, step=0.1)
subsample = st.sidebar.slider("Vector grid subsample", 1, 4, 1, step=1)
vector_mode = st.sidebar.selectbox("Vector mode", ["stream", "quiver"], index=0)
quantize_bins = st.sidebar.slider("Speed quantization bins (0=off)", 0, 12, 0, step=1)

st.sidebar.subheader("Trajectory Bundling")
resample_points = st.sidebar.slider("Control points per trajectory (K)", 5, 60, 20, step=1)
beta = st.sidebar.slider("beta (attraction strength)", 0.0, 1.0, 0.25, step=0.05)
iterations = st.sidebar.slider("Bundling iterations", 1, 60, 25, step=1)
neighbor_radius = st.sidebar.slider("Neighbor radius (fraction of diameter)", 0.01, 0.5, 0.1, step=0.01)
highlight_best = st.sidebar.checkbox("Highlight best trajectory", value=True)
temporal_smooth = st.sidebar.slider("Temporal smoothing (MA window)", 1, 9, 3, step=2)
curve_type = st.sidebar.selectbox("Curve type", ["catmull-rom", "polyline"], index=0)
n_clusters = st.sidebar.slider("Clusters (bundling color groups)", 1, 8, 3, step=1)
max_traj = st.sidebar.slider("Max trajectories rendered", 10, 300, 120, step=10)

if st.sidebar.button("Run / Refresh"):
    st.session_state["run"] = True

if "run" not in st.session_state:
    st.info("Set the parameters and click **Run / Refresh**.")
    st.stop()


# -----------------------------------------------------------------------------
# Cached execution
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def cached_run(pop_size, n_generations, hidden_dim, mutation_rate, seed):
    return run_evolution(pop_size, n_generations, hidden_dim, mutation_rate, seed)


@st.cache_data(show_spinner=False)
def cached_alignment(weights_by_gen, lambda_align, seed):
    return compute_aligned_umap_embedding(weights_by_gen, lambda_align=lambda_align, random_state=seed)


evolution = cached_run(pop_size, n_generations, hidden_dim, mutation_rate, seed)
embedding_all, gen_labels, per_gen_embeddings = cached_alignment(
    evolution.weights_by_gen, lambda_align, seed
)

fitness_cmap = get_discrete_cmap(fitness_cmap_name, n=20)
gen_cmap = get_discrete_cmap(gen_cmap_name, n=20)
fitness_cmap_bins = fitness_bins


st.markdown("---")
st.subheader("Aligned UMAP")
try:
    fig = plot_aligned_umap(
        evolution.weights_by_gen,
        evolution.fitness_by_gen,
        lambda_align=lambda_align,
        random_state=seed,
        cmap_gen=gen_cmap,
        cmap_fit=fitness_cmap,
        norm_mode=norm_mode,
        gamma=gamma,
        fitness_bins=fitness_cmap_bins,
    )
    st.pyplot(fig, clear_figure=True)
except Exception as exc:  # noqa: BLE001
    st.error(f"Error while generating Aligned UMAP: {exc}")

st.markdown("---")
st.subheader("Vector Field")
try:
    fig = plot_vector_field(
        evolution.weights_by_gen,
        evolution.fitness_by_gen,
        lambda_align=lambda_align,
        grid_res=grid_res,
        random_state=seed,
        show_points=show_points,
        normalize_vectors=normalize_vectors,
        smoothing_sigma=smoothing_sigma,
        subsample=subsample,
        vector_mode=vector_mode,
        quantize_bins=quantize_bins if quantize_bins > 0 else None,
        cmap_gen=gen_cmap,
        cmap_fit=fitness_cmap,
        norm_mode=norm_mode,
        gamma=gamma,
        fitness_bins=fitness_cmap_bins,
    )
    st.pyplot(fig, clear_figure=True)
except Exception as exc:  # noqa: BLE001
    st.error(f"Error while generating Vector Field: {exc}")

st.markdown("---")
st.subheader("Trajectory Bundling")
try:
    all_pts = np.vstack(per_gen_embeddings) if per_gen_embeddings else np.empty((0, 2))
    diag = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0)) if len(all_pts) else 1.0
    radius_abs = neighbor_radius * diag

    fig = plot_trajectory_bundling(
        evolution.weights_by_gen,
        evolution.fitness_by_gen,
        lambda_align=lambda_align,
        beta=beta,
        iterations=iterations,
        resample_points=resample_points,
        neighbor_radius=radius_abs,
        random_state=seed,
        highlight_best=highlight_best,
        temporal_smooth=temporal_smooth,
        curve_type=curve_type,
        n_clusters=n_clusters,
        max_trajectories=max_traj,
        norm_mode=norm_mode,
        gamma=gamma,
        cmap=fitness_cmap,
        fitness_bins=fitness_cmap_bins,
    )
    st.pyplot(fig, clear_figure=True)
except Exception as exc:  # noqa: BLE001
    st.error(f"Error while generating Trajectory Bundling: {exc}")


st.markdown(
    """
Quick notes:
- **Aligned UMAP:** 2D projection per generation with temporal post-alignment `proj_aligned = proj_k - lambda * (proj_k - proj_ref)`.
- **Vector Field:** representation-flow field using velocities `v_i(t+1) - v_i(t)` aggregated on a regular grid and rendered with `plt.streamplot`.
- **Trajectory Bundling:** resampled trajectories, iterative attraction `p_i_new = p_i + beta * sum (p_j - p_i)` among neighbors, rendered with `alpha=0.4`.
"""
)
