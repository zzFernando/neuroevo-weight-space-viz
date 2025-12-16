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
from visualizations.vector_field import plot as plot_vector_field
from visualizations.vector_field import plot_interactive as plot_vector_field_interactive


st.set_page_config(page_title="Neuroevo Weight-Space Viz", layout="wide")

TEXTS = {
    "pt": {
        "lang_label": "Idioma / Language",
        "lang_names": {"pt": "Português", "en": "English"},
        "title": "Visualizing Weight-Space Dynamics in Neuroevolution using Aligned Projections and Vector Fields",
        "intro": (
            "Ferramenta para visualizar como uma população de redes neurais evolui no espaço de pesos, "
            "usando projeções alinhadas, campos vetoriais e agrupamento de trajetórias."
        ),
        "sidebar_run_header": "Execução / Evolução",
        "pop_size": "Tamanho da população",
        "pop_size_help": "Número de indivíduos (redes) por geração.",
        "n_generations": "Número de gerações",
        "n_generations_help": "Quantas gerações serão simuladas nesta execução.",
        "hidden_dim": "Neurônios ocultos",
        "hidden_dim_help": "Tamanho da camada oculta do MLP avaliado em cada indivíduo.",
        "mutation_rate": "Taxa de mutação (ruído)",
        "mutation_rate_help": "Intensidade do ruído gaussiano aplicado aos pesos durante a mutação.",
        "seed": "Semente aleatória",
        "seed_help": "Semente para reprodutibilidade dos resultados e das projeções.",
        "align_header": "Alinhamento e renderização",
        "lambda_align": "Lambda de alinhamento",
        "lambda_align_help": "Peso do alinhamento temporal entre projeções (0 = sem alinhamento; 1 = máximo).",
        "render_mode": "Renderização",
        "render_mode_help": "Interativa (Plotly) permite zoom/hover; Estática (Matplotlib) é mais leve.",
        "render_interactive": "Interativa (Plotly)",
        "render_static": "Estática (Matplotlib)",
        "run_button": "Executar / Atualizar",
        "run_caption": "Use os controles globais e clique para gerar as visualizações.",
        "run_info": "Defina os parâmetros e clique em **Executar / Atualizar**.",
        "tabs": ["Aligned UMAP", "Vector Field"],
        "umap_sub": "Aligned UMAP",
        "umap_desc": (
            "Mostra uma projeção 2D dos pesos por geração, com alinhamento temporal para reduzir saltos visuais. "
            "Permite ver onde a população se concentra e como deriva ao longo do tempo. "
            "Importa para relacionar regiões do espaço de pesos ao fitness obtido."
        ),
        "umap_caption": "Projeção 2D alinhada por geração; cores indicam geração e fitness.",
        "vector_sub": "Vector Field",
        "vector_desc": (
            "Mostra o campo vetorial do deslocamento médio da população entre gerações em uma grade. "
            "Setas indicam direção e intensidade da mudança no embedding. "
            "Importa para identificar fluxos dominantes no espaço de pesos."
        ),
        "vector_caption": "Campo vetorial médio entre gerações; pontos coloridos por geração e fitness.",
        "quick_notes": "",
    },
    "en": {
        "lang_label": "Idioma / Language",
        "lang_names": {"pt": "Português", "en": "English"},
        "title": "Visualizing Weight-Space Dynamics in Neuroevolution using Aligned Projections and Vector Fields",
        "intro": (
            "Tool to visualize how a population of neural networks evolves in weight space, "
            "using aligned projections, vector fields, and bundled trajectories."
        ),
        "sidebar_run_header": "Run / Evolution",
        "pop_size": "Population size",
        "pop_size_help": "Number of individuals (networks) per generation.",
        "n_generations": "Number of generations",
        "n_generations_help": "How many generations to simulate in this run.",
        "hidden_dim": "Hidden neurons",
        "hidden_dim_help": "Hidden layer size of the MLP evaluated for each individual.",
        "mutation_rate": "Mutation rate (noise)",
        "mutation_rate_help": "Intensity of the Gaussian noise applied to weights during mutation.",
        "seed": "Random seed",
        "seed_help": "Seed for reproducibility of results and projections.",
        "align_header": "Alignment & rendering",
        "lambda_align": "Alignment lambda",
        "lambda_align_help": "Weight of the temporal alignment between projections (0 = no alignment; 1 = strongest).",
        "render_mode": "Rendering",
        "render_mode_help": "Interactive (Plotly) enables zoom/hover; Static (Matplotlib) is lighter.",
        "render_interactive": "Interactive (Plotly)",
        "render_static": "Static (Matplotlib)",
        "run_button": "Run / Refresh",
        "run_caption": "Set the global controls and click to render the visualizations.",
        "run_info": "Choose parameters and click **Run / Refresh**.",
        "tabs": ["Aligned UMAP", "Vector Field"],
        "umap_sub": "Aligned UMAP",
        "umap_desc": (
            "Shows a 2D projection of weights per generation with temporal alignment to reduce visual jumps. "
            "Use it to see where the population concentrates and how it drifts over time. "
            "Important to relate regions of weight space to achieved fitness."
        ),
        "umap_caption": "Aligned 2D projection per generation; colors encode generation and fitness.",
        "vector_sub": "Vector Field",
        "vector_desc": (
            "Summarizes the average movement of the population between generations on a grid. "
            "Arrows indicate direction and strength of change in the embedding. "
            "Useful to spot dominant flows in weight space."
        ),
        "vector_caption": "Average vector field between generations; points colored by generation and fitness.",
        "quick_notes": "",
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
pop_size = st.sidebar.slider(
    t("pop_size"),
    20,
    400,
    120,
    step=20,
    help=t("pop_size_help"),
)
n_generations = st.sidebar.slider(
    t("n_generations"),
    5,
    80,
    25,
    step=1,
    help=t("n_generations_help"),
)
hidden_dim = st.sidebar.slider(
    t("hidden_dim"),
    4,
    128,
    32,
    step=4,
    help=t("hidden_dim_help"),
)
mutation_rate = st.sidebar.slider(
    t("mutation_rate"),
    0.005,
    0.2,
    0.05,
    step=0.005,
    help=t("mutation_rate_help"),
)
seed = st.sidebar.number_input(
    t("seed"),
    min_value=0,
    max_value=9999,
    value=42,
    help=t("seed_help"),
)

st.sidebar.subheader(t("align_header"))
lambda_align = st.sidebar.slider(
    t("lambda_align"),
    0.0,
    1.0,
    0.3,
    step=0.05,
    help=t("lambda_align_help"),
)
render_mode = st.sidebar.radio(
    t("render_mode"),
    [t("render_interactive"), t("render_static")],
    index=1,
    help=t("render_mode_help"),
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

tab_umap, tab_vector = st.tabs(t("tabs"))

# Paletas fixas (mesmas cores usadas nos slides)
CMAP_GEN = plt.cm.plasma
CMAP_FIT_UMAP = get_discrete_cmap("cividis", n=20)
CMAP_FIT_VEC = get_discrete_cmap("cividis", n=20)

# -----------------------------------------------------------------------------
# Tab: Aligned UMAP
# -----------------------------------------------------------------------------
with tab_umap:
    st.subheader(t("umap_sub"))
    st.markdown(t("umap_desc"))
    st.caption(t("umap_caption"))
    norm_mode_umap = "power"
    gamma_umap = 0.3
    fitness_bins_umap = 9

    try:
        gen_cmap = CMAP_GEN
        fitness_cmap = CMAP_FIT_UMAP
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
    st.caption(t("vector_caption"))
    grid_res = 22
    smoothing_sigma = 1.0
    subsample = 1
    vector_mode = "stream"
    show_points = True
    normalize_vectors = False
    quantize_bins = 0

    try:
        gen_cmap = CMAP_GEN
        fitness_cmap = CMAP_FIT_VEC
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

if t("quick_notes"):
    st.markdown(t("quick_notes"))

