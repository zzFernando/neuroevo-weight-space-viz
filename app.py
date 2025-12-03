# app.py
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils import compute_aligned_umap_embedding, run_evolution
from visualizations.aligned_umap import plot as plot_aligned_umap
from visualizations.trajectory_bundling import plot as plot_trajectory_bundling
from visualizations.vector_field import plot as plot_vector_field

st.set_page_config(page_title="Neuroevo Weight-Space Viz", layout="wide")

st.title("Exploring Neuroevolution Weight Space (Cantareira 2020)")
st.markdown(
    """
    ImplementaÇõÇœes fiÇüs das trÇës visualizaÇõÇæes descritas em
    **Cantareira, Etemad & Paulovich (2020)** para explorar trajetÇürias no
    espaÇõo de representaÇõÇões:
    **Aligned UMAP**, **Vector Field** e **Trajectory Bundling**.
    """
)


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header("ParÇ½metros da NeuroevoluÇõÇœo")
pop_size = st.sidebar.slider("Tamanho da populaÇõÇœo", 20, 400, 120, step=20)
n_generations = st.sidebar.slider("NÇ§mero de geraÇõÇæes", 5, 80, 25, step=1)
hidden_dim = st.sidebar.slider("NeurÇïnios na camada oculta", 4, 128, 32, step=4)
mutation_rate = st.sidebar.slider("Taxa de mutaÇõÇœo", 0.005, 0.2, 0.05, step=0.005)
seed = st.sidebar.number_input("Seed aleatÇürio", min_value=0, max_value=9999, value=42)

st.sidebar.header("VisualizaÇõÇœes")
mode = st.sidebar.selectbox(
    "Visualization Mode",
    ["Aligned UMAP", "Vector Field", "Trajectory Bundling"],
)
lambda_align = st.sidebar.slider("λ (alinhamento UMAP)", 0.0, 1.0, 0.3, step=0.05)

if mode == "Vector Field":
    grid_res = st.sidebar.slider("ResoluÇõÇœ da grade (campo vetorial)", 5, 60, 20, step=1)
    show_points = st.sidebar.checkbox("Mostrar pontos das geraÇõÇœes", value=True)
elif mode == "Trajectory Bundling":
    resample_points = st.sidebar.slider("Pontos de controle por trajetÇüria (K)", 5, 60, 20, step=1)
    beta = st.sidebar.slider("β (forÇõa de atraÇõÇœ)", 0.0, 1.0, 0.35, step=0.05)
    iterations = st.sidebar.slider("IteraÇõÇæes de bundling", 1, 60, 15, step=1)
    neighbor_radius = st.sidebar.slider("Raio de vizinhança (fraÇõÇœ do diÇümetro)", 0.01, 0.5, 0.1, step=0.01)

if st.sidebar.button("Rodar EvoluÇõÇœ / Atualizar"):
    st.session_state["run"] = True

if "run" not in st.session_state:
    st.info("Ajuste os parÇ½metros e clique em **Rodar EvoluÇõÇœ / Atualizar**.")
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


# -----------------------------------------------------------------------------
# Fitness curve for context
# -----------------------------------------------------------------------------
st.subheader("Curva de fitness")
fig_curve, ax = plt.subplots(figsize=(6, 4))
gens = np.arange(len(evolution.mean_fitness))
ax.errorbar(
    gens,
    evolution.mean_fitness,
    yerr=evolution.std_fitness,
    fmt="-o",
    capsize=4,
    linewidth=2,
    markersize=4,
)
ax.set_xlabel("GeraÇõÇœo")
ax.set_ylabel("Fitness mÇ¸dio ± desvio")
ax.set_title("EvoluÇõÇœ do fitness")
fig_curve.tight_layout()
st.pyplot(fig_curve, clear_figure=True)


st.markdown("---")
st.subheader(f"VisualizaÇõÇœ selecionada: {mode}")

try:
    if mode == "Aligned UMAP":
        fig = plot_aligned_umap(
            evolution.weights_by_gen,
            evolution.fitness_by_gen,
            lambda_align=lambda_align,
            random_state=seed,
        )
        st.pyplot(fig, clear_figure=True)
    elif mode == "Vector Field":
        fig = plot_vector_field(
            evolution.weights_by_gen,
            evolution.fitness_by_gen,
            lambda_align=lambda_align,
            grid_res=grid_res,
            random_state=seed,
            show_points=show_points,
        )
        st.pyplot(fig, clear_figure=True)
    elif mode == "Trajectory Bundling":
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
        )
        st.pyplot(fig, clear_figure=True)
except Exception as exc:  # noqa: BLE001
    st.error(f"Erro ao gerar visualizaÇõÇœ: {exc}")


st.markdown(
    """
Notas rÇüpidas:
- **Aligned UMAP:** projeÇõÇœ 2D por geraÇõÇœ com alinhamento temporal pÇüós-processado `proj_aligned = proj_k - λ * (proj_k - proj_ref)`.
- **Vector Field:** campo vetorial do fluxo da representaÇõÇœ usando velocidades `α_i(t+1) - α_i(t)` agregadas em grade regular e renderizadas com `plt.streamplot`.
- **Trajectory Bundling:** reamostragem das trajetÇürias, atraÇõÇœ iterativa `p_i_new = p_i + β * Σ (p_j - p_i)` entre vizinhos e renderizaÇõÇœ com `alpha=0.4`.
"""
)
