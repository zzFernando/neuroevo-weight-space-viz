# app.py
import os

# Evita o backend workqueue do numba (não thread-safe) forçando OMP.
# Também removemos NUMBA_NUM_THREADS herdado do ambiente para evitar conflitos.
os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")
os.environ.pop("NUMBA_NUM_THREADS", None)

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils import compute_aligned_umap_embedding, run_evolution
from visualizations.aligned_umap import plot as plot_aligned_umap
from visualizations.aligned_umap import plot_interactive as plot_aligned_umap_interactive
from visualizations.common import get_discrete_cmap
from visualizations.trajectory_bundling import plot as plot_trajectory_bundling
from visualizations.trajectory_bundling import plot_interactive as plot_trajectory_bundling_interactive
from visualizations.vector_field import plot as plot_vector_field
from visualizations.vector_field import plot_interactive as plot_vector_field_interactive


st.set_page_config(page_title="Neuroevo Weight-Space Viz", layout="wide")

TEXTS = {
    "pt": {
        "lang_label": "Idioma / Language",
        "lang_names": {"pt": "Português", "en": "English"},
        "title": "Explorando o Espaço de Pesos na Neuroevolução",
        "intro": (
            "Ferramenta para explorar trajetórias no espaço de pesos com três visualizações: "
            "**Aligned UMAP**, **Vector Field** e **Trajectory Bundling**."
        ),
        "sidebar_run_header": "Execução / Evolução",
        "pop_size": "Population size",
        "n_generations": "Number of generations",
        "hidden_dim": "Hidden neurons",
        "mutation_rate": "Mutation rate",
        "seed": "Random seed",
        "align_header": "Alinhamento e renderização",
        "lambda_align": "lambda (UMAP alignment)",
        "render_mode": "Rendering",
        "render_interactive": "Interativa (Plotly)",
        "render_static": "Estática (Matplotlib)",
        "run_button": "Run / Refresh",
        "run_caption": "Use as abas abaixo para configurar cada visualização separadamente.",
        "run_info": "Defina os parâmetros globais e clique em **Run / Refresh**.",
        "tabs": ["Aligned UMAP", "Vector Field", "Trajectory Bundling"],
        "umap_sub": "Aligned UMAP",
        "umap_desc": (
            "UMAP por geração com alinhamento temporal (`proj_k_aligned = proj_k - lambda * (proj_k - proj_ref)`). "
            "Útil para ver a deriva das nuvens de pesos ao longo do tempo.\n\n"
            "Parâmetros:\n"
            "- `lambda`: quanto puxar a projeção atual em direção à referência anterior (0 = sem alinhamento).\n"
            "- `Generation/Fitness colormap`: paletas para cor de geração e de fitness.\n"
            "- `Color normalization`: modo de normalização de cores (power, corte ou equalização).\n"
            "- `Power gamma`: curvatura usada no modo power.\n"
            "- `Fitness quantile bins`: número de bins discretos para fitness."
        ),
        "settings": "Configuração",
        "gen_cmap": "Generation colormap",
        "fit_cmap": "Fitness colormap",
        "color_norm": "Color normalization",
        "power_gamma": "Power gamma (fitness)",
        "fitness_bins": "Fitness quantile bins",
        "vector_sub": "Vector Field",
        "vector_desc": (
            "Campo de vetores no espaço UMAP agregando o deslocamento médio entre gerações. "
            "Mostra direção e magnitude do fluxo da população.\n\n"
            "Parâmetros:\n"
            "- `Grid resolution`: número de células para acumular velocidades.\n"
            "- `Vector smoothing (sigma)`: suavização Gaussiana (0 desliga).\n"
            "- `Vector grid subsample`: mostra 1 em N vetores/células.\n"
            "- `Vector mode`: linhas de fluxo (`stream`) ou setas (`quiver`).\n"
            "- `Show generation points`: exibe pontos originais.\n"
            "- `Normalize vectors`: usa apenas direções (módulo unitário).\n"
            "- `Speed quantization bins`: discretiza velocidades (0 = contínuo).\n"
            "- `Generation/Fitness colormap`: paletas para pintar os pontos."
        ),
        "grid_res": "Grid resolution",
        "smooth_sigma": "Vector smoothing (sigma)",
        "subsample": "Vector grid subsample",
        "vector_mode": "Vector mode",
        "show_points": "Show generation points",
        "normalize_vectors": "Normalize vectors (unit length)",
        "quantize_bins": "Speed quantization bins (0=off)",
        "bundle_sub": "Trajectory Bundling",
        "bundle_desc": (
            "Agrupa trajetórias no UMAP alinhado, atraindo curvas semelhantes e podendo suavizar com Catmull-Rom. "
            "Bom para enxergar padrões dominantes.\n\n"
            "Parâmetros:\n"
            "- `Control points per trajectory (K)`: reamostra cada trajetória para K pontos.\n"
            "- `beta`: força de atração entre trajetórias vizinhas.\n"
            "- `Bundling iterations`: quantas iterações de atração aplicar.\n"
            "- `Neighbor radius`: raio de influência (fração do diâmetro do embedding).\n"
            "- `Temporal smoothing`: média móvel temporal antes do bundling.\n"
            "- `Curve type`: `catmull-rom` (suave) ou `polyline` (segmentos).\n"
            "- `Clusters`: número de grupos de cor (k-means); `Max trajectories`: limite de curvas desenhadas.\n"
            "- `Highlight best trajectory`: destaca a de maior fitness médio; `Fitness colormap`: paleta usada."
        ),
        "control_points": "Control points per trajectory (K)",
        "beta": "beta (attraction strength)",
        "iterations": "Bundling iterations",
        "neighbor_radius": "Neighbor radius (fraction of diameter)",
        "highlight_best": "Highlight best trajectory",
        "temporal_smooth": "Temporal smoothing (MA window)",
        "curve_type": "Curve type",
        "n_clusters": "Clusters (bundling color groups)",
        "max_traj": "Max trajectories rendered",
        "quick_notes": (
            "Notas rápidas:\n"
            "- **Aligned UMAP:** projeção 2D por geração com alinhamento temporal.\n"
            "- **Vector Field:** campo de deslocamento médio entre gerações em grade regular.\n"
            "- **Trajectory Bundling:** trajetórias reamostradas e atraídas, opcionalmente suavizadas."
        ),
    },
    "en": {
        "lang_label": "Idioma / Language",
        "lang_names": {"pt": "Português", "en": "English"},
        "title": "Exploring Neuroevolution Weight Space",
        "intro": (
            "Tool to explore trajectories in weight space with three visualizations: "
            "**Aligned UMAP**, **Vector Field**, and **Trajectory Bundling**."
        ),
        "sidebar_run_header": "Run / Evolution",
        "pop_size": "Population size",
        "n_generations": "Number of generations",
        "hidden_dim": "Hidden neurons",
        "mutation_rate": "Mutation rate",
        "seed": "Random seed",
        "align_header": "Alignment & rendering",
        "lambda_align": "lambda (UMAP alignment)",
        "render_mode": "Rendering",
        "render_interactive": "Interactive (Plotly)",
        "render_static": "Static (Matplotlib)",
        "run_button": "Run / Refresh",
        "run_caption": "Use the tabs below to tune each visualization separately.",
        "run_info": "Set global parameters and click **Run / Refresh**.",
        "tabs": ["Aligned UMAP", "Vector Field", "Trajectory Bundling"],
        "umap_sub": "Aligned UMAP",
        "umap_desc": (
            "Per-generation UMAP with temporal alignment (`proj_k_aligned = proj_k - lambda * (proj_k - proj_ref)`). "
            "Great to see how weight clouds drift over time.\n\n"
            "Parameters:\n"
            "- `lambda`: how strongly to pull the current projection toward the previous reference (0 = no alignment).\n"
            "- `Generation/Fitness colormap`: palettes for generation and fitness colors.\n"
            "- `Color normalization`: color scaling mode (power, clipping, or equalized).\n"
            "- `Power gamma`: curvature used in power mode.\n"
            "- `Fitness quantile bins`: number of discrete bins for fitness."
        ),
        "settings": "Settings",
        "gen_cmap": "Generation colormap",
        "fit_cmap": "Fitness colormap",
        "color_norm": "Color normalization",
        "power_gamma": "Power gamma (fitness)",
        "fitness_bins": "Fitness quantile bins",
        "vector_sub": "Vector Field",
        "vector_desc": (
            "Vector field in UMAP space aggregating average displacement between generations. "
            "Shows direction and magnitude of population flow.\n\n"
            "Parameters:\n"
            "- `Grid resolution`: number of cells to accumulate velocities.\n"
            "- `Vector smoothing (sigma)`: Gaussian smoothing (0 turns off).\n"
            "- `Vector grid subsample`: show 1 in N vectors/cells.\n"
            "- `Vector mode`: streamlines (`stream`) or arrows (`quiver`).\n"
            "- `Show generation points`: display original points.\n"
            "- `Normalize vectors`: use directions only (unit length).\n"
            "- `Speed quantization bins`: discretize speeds (0 = continuous).\n"
            "- `Generation/Fitness colormap`: palettes for the points."
        ),
        "grid_res": "Grid resolution",
        "smooth_sigma": "Vector smoothing (sigma)",
        "subsample": "Vector grid subsample",
        "vector_mode": "Vector mode",
        "show_points": "Show generation points",
        "normalize_vectors": "Normalize vectors (unit length)",
        "quantize_bins": "Speed quantization bins (0=off)",
        "bundle_sub": "Trajectory Bundling",
        "bundle_desc": (
            "Bundles trajectories in aligned UMAP, pulling similar curves together and optionally smoothing with "
            "Catmull-Rom. Good to surface dominant patterns.\n\n"
            "Parameters:\n"
            "- `Control points per trajectory (K)`: resample each trajectory to K points.\n"
            "- `beta`: attraction strength between neighboring trajectories.\n"
            "- `Bundling iterations`: how many attraction iterations to run.\n"
            "- `Neighbor radius`: influence radius (fraction of embedding diameter).\n"
            "- `Temporal smoothing`: moving average along time before bundling.\n"
            "- `Curve type`: `catmull-rom` (smooth) or `polyline` (segments).\n"
            "- `Clusters`: number of color groups (k-means); `Max trajectories`: limit of curves rendered.\n"
            "- `Highlight best trajectory`: highlight highest mean-fitness curve; `Fitness colormap`: palette used."
        ),
        "control_points": "Control points per trajectory (K)",
        "beta": "beta (attraction strength)",
        "iterations": "Bundling iterations",
        "neighbor_radius": "Neighbor radius (fraction of diameter)",
        "highlight_best": "Highlight best trajectory",
        "temporal_smooth": "Temporal smoothing (MA window)",
        "curve_type": "Curve type",
        "n_clusters": "Clusters (bundling color groups)",
        "max_traj": "Max trajectories rendered",
        "quick_notes": (
            "Quick notes:\n"
            "- **Aligned UMAP:** 2D projection per generation with temporal alignment.\n"
            "- **Vector Field:** average displacement field between generations on a regular grid.\n"
            "- **Trajectory Bundling:** resampled and attracted trajectories, optionally smoothed."
        ),
    },
}


language = st.sidebar.radio(
    TEXTS["pt"]["lang_label"],
    ["pt", "en"],
    format_func=lambda v: TEXTS["pt"]["lang_names"][v],
    index=1,
)


def t(key: str) -> str:
    return TEXTS[language][key]


st.title(t("title"))
st.markdown(t("intro"))


# -----------------------------------------------------------------------------
# Sidebar controls (apenas parâmetros globais)
# -----------------------------------------------------------------------------
st.sidebar.header(t("sidebar_run_header"))
pop_size = st.sidebar.slider(t("pop_size"), 20, 400, 120, step=20)
n_generations = st.sidebar.slider(t("n_generations"), 5, 80, 25, step=1)
hidden_dim = st.sidebar.slider(t("hidden_dim"), 4, 128, 32, step=4)
mutation_rate = st.sidebar.slider(t("mutation_rate"), 0.005, 0.2, 0.05, step=0.005)
seed = st.sidebar.number_input(t("seed"), min_value=0, max_value=9999, value=42)

st.sidebar.subheader(t("align_header"))
lambda_align = st.sidebar.slider(t("lambda_align"), 0.0, 1.0, 0.3, step=0.05)
render_mode = st.sidebar.radio(
    t("render_mode"),
    [t("render_interactive"), t("render_static")],
    index=1,
)
use_interactive = render_mode.startswith("Interativa") or render_mode.startswith("Interactive")

if st.sidebar.button(t("run_button")):
    st.session_state["run"] = True

st.sidebar.caption(t("run_caption"))

if "run" not in st.session_state:
    st.info(t("run_info"))
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

cmap_options = ["plasma", "inferno", "magma", "viridis", "cividis", "turbo", "fitness_map"]

tab_umap, tab_vector, tab_bundle = st.tabs(t("tabs"))

# -----------------------------------------------------------------------------
# Tab: Aligned UMAP
# -----------------------------------------------------------------------------
with tab_umap:
    st.subheader(t("umap_sub"))
    st.markdown(t("umap_desc"))
    with st.expander(t("settings"), expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            gen_cmap_name_umap = st.selectbox(
                t("gen_cmap"),
                cmap_options,
                index=0,
                key="umap_gen_cmap",
            )
            fitness_cmap_name_umap = st.selectbox(
                t("fit_cmap"),
                cmap_options,
                index=1,
                key="umap_fit_cmap",
            )
        with col2:
            norm_mode_umap = st.selectbox(
                t("color_norm"),
                ["power", "clipped", "equalized"],
                index=0,
                key="umap_norm",
            )
            gamma_umap = st.slider(t("power_gamma"), 0.1, 1.0, 0.3, step=0.05, key="umap_gamma")
            fitness_bins_umap = st.slider(t("fitness_bins"), 5, 15, 9, step=1, key="umap_bins")

    try:
        gen_cmap = get_discrete_cmap(gen_cmap_name_umap, n=20)
        fitness_cmap = get_discrete_cmap(fitness_cmap_name_umap, n=20)
        if use_interactive:
            fig = plot_aligned_umap_interactive(
                evolution.weights_by_gen,
                evolution.fitness_by_gen,
                lambda_align=lambda_align,
                random_state=seed,
                cmap_gen=gen_cmap,
                cmap_fit=fitness_cmap,
                norm_mode=norm_mode_umap,
                gamma=gamma_umap,
                fitness_bins=fitness_bins_umap,
            )
            st.plotly_chart(fig, width="stretch")
        else:
            fig = plot_aligned_umap(
                evolution.weights_by_gen,
                evolution.fitness_by_gen,
                lambda_align=lambda_align,
                random_state=seed,
                cmap_gen=gen_cmap,
                cmap_fit=fitness_cmap,
                norm_mode=norm_mode_umap,
                gamma=gamma_umap,
                fitness_bins=fitness_bins_umap,
            )
            st.pyplot(fig, clear_figure=True)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error while generating Aligned UMAP: {exc}")

# -----------------------------------------------------------------------------
# Tab: Vector Field
# -----------------------------------------------------------------------------
with tab_vector:
    st.subheader(t("vector_sub"))
    st.markdown(t("vector_desc"))
    with st.expander(t("settings"), expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            grid_res = st.slider(t("grid_res"), 5, 60, 20, step=1)
            smoothing_sigma = st.slider(t("smooth_sigma"), 0.0, 3.0, 1.0, step=0.1)
            subsample = st.slider(t("subsample"), 1, 4, 1, step=1)
            vector_mode = st.selectbox(t("vector_mode"), ["stream", "quiver"], index=0)
        with col2:
            show_points = st.checkbox(t("show_points"), value=True)
            normalize_vectors = st.checkbox(t("normalize_vectors"), value=False)
            quantize_bins = st.slider(t("quantize_bins"), 0, 12, 0, step=1)
            gen_cmap_name_vec = st.selectbox(
                t("gen_cmap"),
                cmap_options,
                index=0,
                key="vec_gen_cmap",
            )
            fitness_cmap_name_vec = st.selectbox(
                t("fit_cmap"),
                cmap_options,
                index=1,
                key="vec_fit_cmap",
            )

    try:
        gen_cmap = get_discrete_cmap(gen_cmap_name_vec, n=20)
        fitness_cmap = get_discrete_cmap(fitness_cmap_name_vec, n=20)
        if use_interactive:
            fig = plot_vector_field_interactive(
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
                norm_mode="power",
                gamma=0.3,
                fitness_bins=9,
            )
            st.plotly_chart(fig, width="stretch")
        else:
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
                norm_mode="power",
                gamma=0.3,
                fitness_bins=9,
            )
            st.pyplot(fig, clear_figure=True)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error while generating Vector Field: {exc}")

# -----------------------------------------------------------------------------
# Tab: Trajectory Bundling
# -----------------------------------------------------------------------------
with tab_bundle:
    st.subheader(t("bundle_sub"))
    st.markdown(t("bundle_desc"))
    with st.expander(t("settings"), expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            resample_points = st.slider(t("control_points"), 5, 60, 20, step=1)
            beta = st.slider(t("beta"), 0.0, 1.0, 0.25, step=0.05)
            iterations = st.slider(t("iterations"), 1, 60, 25, step=1)
            neighbor_radius = st.slider(t("neighbor_radius"), 0.01, 0.5, 0.1, step=0.01)
            highlight_best = st.checkbox(t("highlight_best"), value=True)
        with col2:
            temporal_smooth = st.slider(t("temporal_smooth"), 1, 9, 3, step=2)
            curve_type = st.selectbox(t("curve_type"), ["catmull-rom", "polyline"], index=0)
            n_clusters = st.slider(t("n_clusters"), 1, 8, 3, step=1)
            max_traj = st.slider(t("max_traj"), 10, 300, 120, step=10)
            fitness_cmap_name_bundle = st.selectbox(
                t("fit_cmap"),
                cmap_options,
                index=1,
                key="bundle_fit_cmap",
            )

    try:
        all_pts = np.vstack(per_gen_embeddings) if per_gen_embeddings else np.empty((0, 2))
        diag = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0)) if len(all_pts) else 1.0
        radius_abs = neighbor_radius * diag
        fitness_cmap = get_discrete_cmap(fitness_cmap_name_bundle, n=20)

        if use_interactive:
            fig = plot_trajectory_bundling_interactive(
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
                norm_mode="linear",
                gamma=0.3,
                cmap=fitness_cmap,
                fitness_bins=9,
            )
            st.plotly_chart(fig, width="stretch")
        else:
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
                norm_mode="linear",
                gamma=0.3,
                cmap=fitness_cmap,
                fitness_bins=9,
            )
            st.pyplot(fig, clear_figure=True)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error while generating Trajectory Bundling: {exc}")


st.markdown(t("quick_notes"))
