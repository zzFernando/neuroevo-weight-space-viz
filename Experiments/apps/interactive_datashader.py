"""
interactive_datashader.py — Panel + HoloViews + Datashader interactive explorer.

Usage:
    cd /path/to/neuroevo-weight-space-viz/supplementary
    pixi run panel serve ../brax/interactive_datashader.py --show --autoreload

    # With specific port:
    pixi run panel serve ../brax/interactive_datashader.py --show --port 5007
"""
from __future__ import annotations

import sys
from pathlib import Path

import colorcet as cc
import datashader as ds
import holoviews as hv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
from holoviews.operation.datashader import dynspread, rasterize
from scipy.ndimage import uniform_filter

# Make supplementary utilities importable when serving from supplementary/
_SUPP = Path(__file__).resolve().parent.parent / "supplementary"
if str(_SUPP) not in sys.path:
    sys.path.insert(0, str(_SUPP))

from visualizations.vector_field import _compute_velocity_grid  # noqa: E402

matplotlib.use("agg")
hv.extension("bokeh")
pn.extension(sizing_mode="stretch_width")

RUNS_DIR = Path(__file__).parent / "runs"
N_BINS_VF = 22
PLOT_W, PLOT_H = 720, 620


# ── Run discovery ────────────────────────────────────────────────────────────
# Key format: "{algo} · {seed}"  e.g. "stress · 42", "open_es · 42"

import re as _re

_RAW_PAT = _re.compile(r"^(.+)_seed(\d+)$")   # matches {algo}_seed{seed}


def _parse_stem(stem: str) -> tuple[str, str] | None:
    """Extract (algo, seed) from stem like 'open_es_seed42' or 'stress_seed42'."""
    m = _RAW_PAT.match(stem)
    return (m.group(1), m.group(2)) if m else None


def _run_label(algo: str, seed: str) -> str:
    return f"{algo} · {seed}"


def discover_runs() -> dict[str, Path]:
    """
    Returns {label: raw_npz_path} for every run in RUNS_DIR.
    Excludes *_embedding.npz (caches).
    """
    runs: dict[str, Path] = {}
    for p in sorted(RUNS_DIR.glob("*.npz")):
        if p.stem.endswith("_embedding"):
            continue
        parsed = _parse_stem(p.stem)
        if parsed:
            algo, seed = parsed
            runs[_run_label(algo, seed)] = p
    return runs


RUN_MAP: dict[str, Path] = discover_runs()  # label → raw .npz


def _embedding_cache_path(raw_path: Path) -> Path:
    return raw_path.with_name(raw_path.stem + "_embedding.npz")


def ensure_embedding(raw_path: Path, lambda_align: float = 0.8) -> Path:
    """
    Returns path to embedding cache, computing it via joint UMAP if needed.
    Cache stores Z_aligned (G,P,2), fitnesses_flat, generations_flat.
    """
    cache = _embedding_cache_path(raw_path)
    if cache.exists():
        return cache

    print(f"  Computing joint UMAP for {raw_path.name} …")
    d = np.load(raw_path, allow_pickle=True)
    populations = d["populations"]      # (G, P, D)
    fitnesses   = d["fitnesses"]        # (G, P)
    G, P, D     = populations.shape

    flat = populations.reshape(G * P, D).astype(np.float32)
    if D > 5_000:
        from sklearn.decomposition import PCA
        flat = PCA(n_components=50, random_state=0).fit_transform(flat)

    import umap as _umap
    Z_flat = _umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                        random_state=0).fit_transform(flat).astype(np.float32)

    Z = Z_flat.reshape(G, P, 2)
    Z_ref = Z[0]
    Z_aligned = ((1 - lambda_align) * Z + lambda_align * Z_ref[None]).astype(np.float32)

    gens_flat = np.repeat(np.arange(G, dtype=np.int32), P)
    fits_flat = fitnesses.reshape(-1).astype(np.float32)

    np.savez_compressed(cache, Z_aligned=Z_aligned,
                        fitnesses_flat=fits_flat, generations_flat=gens_flat)
    print(f"  Cached → {cache.name}")
    return cache


def load_run(label: str, lambda_align: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Load (df, Z_aligned (G,P,2)) for a run label. Computes UMAP cache if missing."""
    raw_path = RUN_MAP[label]
    emb_path = ensure_embedding(raw_path, lambda_align)

    d = np.load(emb_path)
    Z_aligned = d["Z_aligned"].copy()
    fits_flat  = d["fitnesses_flat"]
    gens_flat  = d["generations_flat"]
    G, P, _    = Z_aligned.shape

    if "Z_flat" in d.files:
        Z_raw = d["Z_flat"].reshape(G, P, 2)
        Z_ref = Z_raw[0]
        Z_aligned = ((1 - lambda_align) * Z_raw + lambda_align * Z_ref[None]).astype(np.float32)

    xy      = Z_aligned.reshape(-1, 2)
    algo    = label.split(" · ")[0]
    gen_idx = np.repeat(np.arange(G, dtype=int), P)
    ind_idx = np.tile(np.arange(P, dtype=int), G)
    return pd.DataFrame({
        "x":          xy[:, 0].astype(float),
        "y":          xy[:, 1].astype(float),
        "generation": gens_flat.astype(float),
        "fitness":    fits_flat.astype(float),
        "algo":       algo,
        "label":      label,
        "point_idx":  np.arange(len(xy), dtype=int),
        "gen_idx":    gen_idx,
        "ind_idx":    ind_idx,
    }), Z_aligned


def _has_dynamic_lambda() -> bool:
    return any(
        "Z_flat" in np.load(_embedding_cache_path(p)).files
        for p in RUN_MAP.values()
        if _embedding_cache_path(p).exists()
    )


# ── Paper-style matplotlib render (scatter + streamplot) ──────────────────────

def render_paper_style(
    Zs: dict[str, np.ndarray],
    gen_range: tuple[int, int],
    seed_names: list[str],
) -> pn.pane.Matplotlib:
    """Replicates plot_compact_vector_field from experiments/shared.py."""
    n = len(Zs)
    fig, axes = plt.subplots(
        1, n,
        figsize=(5 * n, 4.5),
        dpi=130,
        squeeze=False,
    )

    for col, (s, Z) in enumerate(Zs.items()):
        ax = axes[0, col]
        g0, g1 = int(gen_range[0]), int(gen_range[1])
        Z_sub = Z[g0 : g1 + 1]          # (G_sub, P, 2)
        if len(Z_sub) < 2:
            ax.text(0.5, 0.5, "need ≥2 gens", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            continue

        emb_all = Z_sub.reshape(-1, 2)
        per_gen = [Z_sub[g] for g in range(len(Z_sub))]

        ax.set_facecolor("white")
        ax.scatter(
            emb_all[:, 0], emb_all[:, 1],
            s=2, alpha=0.12, color="#777777",
            edgecolors="none", zorder=1,
        )

        try:
            Xc, Yc, U, V, speed = _compute_velocity_grid(
                per_gen, grid_res=N_BINS_VF, min_vectors_per_cell=1
            )
            U_f = np.ma.filled(U, 0.0).astype(float)
            V_f = np.ma.filled(V, 0.0).astype(float)
            # 3×3 box smooth (matches shared.py _smooth2d)
            from scipy.ndimage import convolve
            k = np.ones((3, 3), float) / 9.0
            U_f = convolve(U_f, k, mode="nearest")
            V_f = convolve(V_f, k, mode="nearest")
            speed_s = np.sqrt(U_f ** 2 + V_f ** 2)

            if speed_s.max() > 0:
                ax.streamplot(
                    Xc, Yc, U_f, V_f,
                    color=speed_s,
                    cmap=plt.cm.plasma,
                    density=1.2,
                    linewidth=1.0,
                    arrowsize=1.0,
                    zorder=2,
                )
        except Exception as e:
            ax.text(0.5, 0.5, f"VF error\n{e}", ha="center", va="center",
                    fontsize=7, color="#900")

        ax.set_title(s, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("auto")

    fig.suptitle(
        f"HalfCheetah · gens {gen_range[0]}–{gen_range[1]} · paper style",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    mpl_pane = pn.pane.Matplotlib(fig, tight=True, format="png", sizing_mode="stretch_width")
    plt.close(fig)
    return mpl_pane


# ── Subplots: one panel per run, colored by generation ───────────────────────

_ALGO_COLORS = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#795548", "#607D8B",
]

# ── MLP graph renderer (Bokeh, no matplotlib) ─────────────────────────────────

def _mlp_bokeh(weights_flat: np.ndarray,
               obs_dim: int = 17, hidden_dim: int = 16, act_dim: int = 6):
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource

    # parse weights
    i = 0
    W1 = weights_flat[i : i + obs_dim * hidden_dim].reshape(hidden_dim, obs_dim); i += obs_dim * hidden_dim
    b1 = weights_flat[i : i + hidden_dim]; i += hidden_dim
    W2 = weights_flat[i : i + hidden_dim * act_dim].reshape(act_dim, hidden_dim); i += hidden_dim * act_dim
    b2 = weights_flat[i : i + act_dim]

    # node positions  (layer_x, node_y)
    layers = [
        ("in",  obs_dim,    0.0, ["obs_%d" % j for j in range(obs_dim)]),
        ("hid", hidden_dim, 1.5, ["h_%d"   % j for j in range(hidden_dim)]),
        ("out", act_dim,    3.0, ["act_%d" % j for j in range(act_dim)]),
    ]
    node_x, node_y, node_color, node_label = [], [], [], []
    layer_cols = {"in": "#90CAF9", "hid": "#A5D6A7", "out": "#FFCC80"}
    positions: list[tuple[float, float]] = []
    for lname, n, lx, labels in layers:
        for j in range(n):
            y = (n - 1) / 2.0 - j
            positions.append((lx, y))
            node_x.append(lx); node_y.append(y)
            node_color.append(layer_cols[lname])
            node_label.append(labels[j])

    in_end  = obs_dim
    hid_end = obs_dim + hidden_dim

    # edges
    xs, ys, colors, widths = [], [], [], []
    w_max = max(np.abs(W1).max(), np.abs(W2).max()) + 1e-9

    for hi in range(hidden_dim):
        px1, py1 = positions[in_end + hi]
        for ii in range(obs_dim):
            px0, py0 = positions[ii]
            w = float(W1[hi, ii])
            xs.append([px0, px1]); ys.append([py0, py1])
            colors.append("#1565C0" if w >= 0 else "#B71C1C")
            widths.append(max(0.3, 3.5 * abs(w) / w_max))

    for oi in range(act_dim):
        px1, py1 = positions[hid_end + oi]
        for hi in range(hidden_dim):
            px0, py0 = positions[in_end + hi]
            w = float(W2[oi, hi])
            xs.append([px0, px1]); ys.append([py0, py1])
            colors.append("#1565C0" if w >= 0 else "#B71C1C")
            widths.append(max(0.3, 3.5 * abs(w) / w_max))

    p = figure(
        width=380, height=520,
        toolbar_location=None,
        x_range=(-0.3, 3.3), y_range=(-(max(obs_dim, hidden_dim) / 2 + 0.5),
                                        max(obs_dim, hidden_dim) / 2 + 0.5),
        background_fill_color="#FAFAFA",
        title="MLP weights",
    )
    p.title.text_font_size = "11px"
    p.axis.visible = False
    p.grid.visible = False
    p.outline_line_color = None

    edge_src = ColumnDataSource(dict(xs=xs, ys=ys, lc=colors, lw=widths))
    p.multi_line("xs", "ys", line_color="lc", line_width="lw",
                 line_alpha=0.55, source=edge_src)

    node_src = ColumnDataSource(dict(x=node_x, y=node_y,
                                     color=node_color, label=node_label))
    p.circle("x", "y", size=10, color="color", line_color="#555",
             line_width=0.8, source=node_src)
    p.text("x", "y", text="label", source=node_src,
           text_font_size="7px", text_align="center",
           text_baseline="middle", text_color="#333")

    # layer labels
    for lname, _, lx, _ in layers:
        p.text(x=[lx], y=[max(obs_dim, hidden_dim) / 2 + 0.2],
               text=[lname], text_font_size="9px",
               text_align="center", text_color="#666")
    return p


# ── Network inspector state ───────────────────────────────────────────────────

_net_label_store:  str | None = None
_net_gen_store:    int        = 0
_net_ind_store:    int        = 0

_net_trigger  = pn.widgets.IntInput(value=0, visible=False)
_net_info     = pn.pane.Markdown("_Clique num ponto no scatter_",
                                  styles={"font-size": "11px", "color": "#666"})
_net_pane     = pn.pane.Bokeh(sizing_mode="stretch_width")


def _update_network(label: str, gen: int, ind: int) -> None:
    global _net_label_store, _net_gen_store, _net_ind_store
    _net_label_store = label
    _net_gen_store   = gen
    _net_ind_store   = ind
    raw = np.load(RUN_MAP[label], allow_pickle=True)
    pops    = raw["populations"]        # (G, P, D)
    fit_val = float(raw["fitnesses"][gen, ind])
    weights = pops[gen, ind]            # (390,)
    _net_pane.object  = _mlp_bokeh(weights)
    _net_info.object  = f"**{label}** · gen {gen} · ind {ind} · fit {fit_val:.1f}"
    _net_trigger.value += 1


# ── HoloViews velocity field overlay ─────────────────────────────────────────

def make_vector_field(Z: np.ndarray) -> hv.VectorField:
    G, P, _ = Z.shape
    if G < 2:
        return hv.VectorField([]).opts(hv.opts.VectorField())

    pts = Z.reshape(-1, 2)
    pad = 0.05 * float(np.ptp(pts[:, 0]))
    xe  = np.linspace(pts[:, 0].min() - pad, pts[:, 0].max() + pad, N_BINS_VF + 1)
    ye  = np.linspace(pts[:, 1].min() - pad, pts[:, 1].max() + pad, N_BINS_VF + 1)

    U = np.zeros((N_BINS_VF, N_BINS_VF), np.float32)
    V = np.zeros((N_BINS_VF, N_BINS_VF), np.float32)
    C = np.zeros((N_BINS_VF, N_BINS_VF), np.float32)

    for g in range(G - 1):
        z0, z1 = Z[g], Z[g + 1]
        mid    = 0.5 * (z0 + z1)
        ix     = np.clip(np.searchsorted(xe, mid[:, 0]) - 1, 0, N_BINS_VF - 1)
        iy     = np.clip(np.searchsorted(ye, mid[:, 1]) - 1, 0, N_BINS_VF - 1)
        dv     = z1 - z0
        np.add.at(U, (iy, ix), dv[:, 0])
        np.add.at(V, (iy, ix), dv[:, 1])
        np.add.at(C, (iy, ix), 1)

    mask = C > 0
    U[mask] /= C[mask]
    V[mask] /= C[mask]
    U = uniform_filter(U, 3)
    V = uniform_filter(V, 3)

    xc, yc = (
        0.5 * (xe[:-1] + xe[1:]),
        0.5 * (ye[:-1] + ye[1:]),
    )
    Xc, Yc = np.meshgrid(xc, yc)
    mag    = np.sqrt(U ** 2 + V ** 2)
    fmask  = mag.ravel() > 0

    return hv.VectorField(
        (Xc.ravel()[fmask], Yc.ravel()[fmask],
         np.arctan2(V, U).ravel()[fmask],
         mag.ravel()[fmask]),
        kdims=["x", "y"], vdims=["angle", "magnitude"],
    ).opts(
        hv.opts.VectorField(
            color="magnitude", cmap="magma",
            magnitude="magnitude", scale=0.35,
            line_width=1.5, alpha=0.85,
        )
    )


# ── Widgets ───────────────────────────────────────────────────────────────────

all_labels    = sorted(RUN_MAP.keys())
dynamic_lambda = _has_dynamic_lambda()

# Group labels by algo for the header hint
_algos = sorted({lbl.split(" · ")[0] for lbl in all_labels})
_algo_hint = "  |  ".join(_algos) if _algos else "no runs found"

run_widget = pn.widgets.CheckBoxGroup(
    name="Runs", options=all_labels, value=all_labels[:1],
)
color_widget = pn.widgets.RadioButtonGroup(
    name="Color by",
    options=["subplots", "density", "generation", "fitness", "paper style"],
    value="subplots",
    button_type="default",
)
velocity_toggle = pn.widgets.Toggle(
    name="Velocity field overlay", value=False,
    button_type="success", width=200,
)
lambda_widget = pn.widgets.FloatSlider(
    name="λ (alignment)" + ("" if dynamic_lambda else " [fixed]"),
    start=0.0, end=1.0, step=0.05, value=0.8,
    disabled=not dynamic_lambda,
)

_g_max = 0
if RUN_MAP:
    _first_emb = _embedding_cache_path(next(iter(RUN_MAP.values())))
    if _first_emb.exists():
        _g_max = int(np.load(_first_emb)["generations_flat"].max())

gen_slider = pn.widgets.RangeSlider(
    name="Generations", start=0, end=_g_max, value=(0, _g_max), step=1,
)

info_pane = pn.pane.Markdown(
    "_no data loaded_",
    styles={"font-size": "12px", "color": "#666"},
)

CMAPS = {"density": cc.fire, "generation": cc.bmy, "fitness": cc.CET_L19}


# ── Reactive plot ─────────────────────────────────────────────────────────────

def build_plot(runs, lambda_align, color_by, show_vel, gen_range):
    if not runs:
        return pn.pane.HTML(
            "<div style='padding:80px 40px;color:#999;text-align:center;"
            "font-size:16px'>Select at least one run</div>"
        )

    dfs_map: dict[str, pd.DataFrame] = {}
    Zs: dict[str, np.ndarray] = {}
    for label in runs:
        if label not in RUN_MAP:
            continue
        try:
            df, Z = load_run(label, lambda_align)
        except Exception as e:
            return pn.pane.HTML(
                f"<div style='padding:40px;color:#c00'>Error loading {label}:<br>{e}</div>"
            )
        df = df[
            (df.generation >= gen_range[0]) & (df.generation <= gen_range[1])
        ].copy()
        dfs_map[label] = df
        Zs[label] = Z

    if not dfs_map:
        return pn.pane.HTML("<div style='padding:40px;color:#999'>No data</div>")

    # ── Subplots: interactive HoloViews Layout, one panel per run ────────────
    if color_by == "subplots":
        plots = []
        for label, df in dfs_map.items():
            pts = hv.Points(df, kdims=["x", "y"],
                            vdims=["generation", "fitness", "gen_idx", "ind_idx"])

            sel = hv.streams.Selection1D(source=pts)
            _label_capture = label   # closure capture

            def _on_sel(index, _df=df, _label=_label_capture):
                if index:
                    row = _df.iloc[index[0]]
                    _update_network(_label, int(row["gen_idx"]), int(row["ind_idx"]))

            sel.param.watch(lambda e, _cb=_on_sel: _cb(e.new), "index")

            if len(df) > 50_000:
                rast = dynspread(rasterize(pts, aggregator=ds.mean("generation")),
                                 threshold=0.5, max_px=6)
                p = rast.opts(hv.opts.Image(
                    width=420, height=370, bgcolor="white",
                    colorbar=True, cmap="plasma", title=label,
                    colorbar_opts={"title": "gen"},
                    tools=["hover", "box_zoom", "wheel_zoom", "reset", "pan"],
                    active_tools=["wheel_zoom"],
                    xlabel="UMAP 1", ylabel="UMAP 2",
                ))
            else:
                p = pts.opts(hv.opts.Points(
                    width=420, height=370, bgcolor="white",
                    color="generation", cmap="plasma",
                    size=4, alpha=0.5, line_width=0,
                    colorbar=True, title=label,
                    colorbar_opts={"title": "gen"},
                    tools=["tap", "hover", "box_zoom", "wheel_zoom", "reset", "pan"],
                    active_tools=["tap"],
                    xlabel="UMAP 1", ylabel="UMAP 2",
                ))
            plots.append(p)
        ncols = min(len(plots), 3)
        return hv.Layout(plots).cols(ncols)

    # ── Paper style: matplotlib scatter + streamplot ──────────────────────────
    if color_by == "paper style":
        return render_paper_style(Zs, gen_range, list(Zs.keys()))

    # ── Datashader modes ── warn if mixing different UMAP spaces ─────────────
    algos_selected = {lbl.split(" · ")[0] for lbl in Zs}
    mixed_warning = ""
    if len(algos_selected) > 1:
        mixed_warning = (
            "<div style='background:#fff3cd;border:1px solid #ffc107;"
            "padding:6px 12px;margin-bottom:8px;border-radius:4px;"
            "font-size:12px;color:#856404'>"
            "⚠ Algoritmos diferentes têm espaços UMAP independentes — "
            "coordenadas não são comparáveis entre runs. Use <b>subplots</b>.</div>"
        )

    # ── Datashader modes ──────────────────────────────────────────────────────
    df_all = pd.concat(list(dfs_map.values()), ignore_index=True)
    info_pane.object = (
        f"**{len(df_all):,}** individuals · "
        f"gens {int(df_all.generation.min())}–{int(df_all.generation.max())} · "
        f"fitness {df_all.fitness.min():.1f}–{df_all.fitness.max():.1f}"
    )

    points = hv.Points(df_all, kdims=["x", "y"], vdims=["generation", "fitness"])
    cmap   = CMAPS[color_by]

    base_opts = dict(
        width=PLOT_W, height=PLOT_H, bgcolor="white",
        colorbar=True, cmap=cmap,
        tools=["hover", "box_zoom", "wheel_zoom", "reset", "pan"],
        active_tools=["wheel_zoom"],
        xlabel="UMAP 1", ylabel="UMAP 2",
    )

    if color_by == "density":
        base_opts.update(cnorm="log", colorbar_opts={"title": "count"})
        rast = dynspread(rasterize(points), threshold=0.5, max_px=4)
    elif color_by == "generation":
        base_opts["colorbar_opts"] = {"title": "generation"}
        rast = dynspread(rasterize(points, aggregator=ds.mean("generation")),
                         threshold=0.5, max_px=4)
    else:
        base_opts["colorbar_opts"] = {"title": "fitness"}
        rast = dynspread(rasterize(points, aggregator=ds.mean("fitness")),
                         threshold=0.5, max_px=4)

    plot = rast.opts(hv.opts.Image(**base_opts))

    if show_vel:
        g0, g1 = int(gen_range[0]), int(gen_range[1])
        layers = [plot]
        for s, Z in Zs.items():
            Z_sub = Z[g0 : g1 + 1]
            if len(Z_sub) >= 2:
                layers.append(make_vector_field(Z_sub))
        hv_plot = hv.Overlay(layers)
    else:
        hv_plot = plot

    if mixed_warning:
        return pn.Column(
            pn.pane.HTML(mixed_warning),
            pn.panel(hv_plot),
        )
    return hv_plot


plot_panel = pn.bind(
    build_plot,
    runs=run_widget,
    lambda_align=lambda_widget,
    color_by=color_widget,
    show_vel=velocity_toggle,
    gen_range=gen_slider,
)


# ── Layout ────────────────────────────────────────────────────────────────────

sidebar = pn.Column(
    pn.pane.Markdown(f"### Runs\n_{_algo_hint}_"),
    run_widget,
    pn.layout.Divider(),
    pn.pane.Markdown("### Display"),
    color_widget,
    velocity_toggle,
    pn.layout.Divider(),
    pn.pane.Markdown("### Parameters"),
    lambda_widget,
    gen_slider,
    pn.layout.Divider(),
    info_pane,
    width=260,
    margin=(10, 10),
)

net_col = pn.Column(
    pn.pane.Markdown("### Network Inspector"),
    _net_info,
    _net_pane,
    width=400,
    margin=(10, 6),
    styles={"border-left": "1px solid #e0e0e0", "padding-left": "10px"},
)

main_col = pn.Column(
    pn.pane.Markdown("## Neuroevolution · Weight Space Explorer", margin=(5, 0, 8, 0)),
    pn.panel(plot_panel, loading_indicator=True),
    sizing_mode="stretch_width",
)

app = pn.Row(sidebar, main_col, net_col, sizing_mode="stretch_width")
app.servable(title="Neuroevo Weight Space Viz")

if __name__ == "__main__":
    pn.serve(app, show=True, port=5007)
