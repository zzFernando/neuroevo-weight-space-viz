"""Generate a pipeline diagram for the paper (Figure 1)."""
from __future__ import annotations
import sys
from pathlib import Path
from paths import PAPER_FIGURES

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
import numpy as np

fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=150)
fig.patch.set_facecolor("white")

BLUE   = "#264653"
TEAL   = "#2a9d8f"
ORANGE = "#e9c46a"
RED    = "#e76f51"
GRAY   = "#adb5bd"
LIGHT  = "#f8f9fa"

# ── Panel 1: Population in weight space ──────────────────────────────────
ax = axes[0]
ax.set_facecolor(LIGHT)
rng = np.random.default_rng(42)
n = 60
for g, (col, alpha) in enumerate([(GRAY, 0.4), (TEAL, 0.6), (BLUE, 1.0)]):
    pts = rng.normal([g * 0.6, g * 0.3], 0.6, (n, 2))
    ax.scatter(pts[:, 0], pts[:, 1], c=col, s=18, alpha=alpha, edgecolors="none")
ax.set_title("1. Evolving population\n(weight space, high-D)", fontsize=10, fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("$d_w$ dimensions (projected)", fontsize=8)
ax.spines[["top","right","left","bottom"]].set_visible(False)
ax.text(0.5, -0.18, "Gen 0 → Gen T", transform=ax.transAxes,
        ha="center", fontsize=8, color=GRAY)

# ── Panel 2: UMAP joint embedding ────────────────────────────────────────
ax = axes[1]
ax.set_facecolor(LIGHT)
colors = plt.cm.turbo(np.linspace(0.1, 0.9, 5))
for g in range(5):
    angle = g * 0.3
    c = [np.cos(angle) * 0.5, np.sin(angle) * 0.5]
    pts = rng.normal(c, 0.25, (n, 2))
    ax.scatter(pts[:, 0], pts[:, 1], color=colors[g], s=14, alpha=0.7, edgecolors="none")
ax.set_title("2. Joint UMAP\n(2D embedding)", fontsize=10, fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])
ax.spines[["top","right","left","bottom"]].set_visible(False)
sm = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(0, 4))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label("generation", fontsize=7)
cbar.set_ticks([0, 4]); cbar.set_ticklabels(["0", "T"])

# ── Panel 3: Alignment ───────────────────────────────────────────────────
ax = axes[2]
ax.set_facecolor(LIGHT)
ref = rng.normal([0, 0], 0.25, (n, 2))
unaligned = rng.normal([1.2, 0.8], 0.28, (n, 2))
aligned = unaligned * 0.2 + ref * 0.8 + rng.normal(0, 0.05, unaligned.shape)
ax.scatter(ref[:, 0], ref[:, 1], color=colors[0], s=14, alpha=0.6, edgecolors="none", label="Gen 0 (ref)")
ax.scatter(unaligned[:, 0], unaligned[:, 1], color=GRAY, s=10, alpha=0.4, edgecolors="none", label="Gen $k$ (raw)")
ax.scatter(aligned[:, 0], aligned[:, 1], color=colors[4], s=14, alpha=0.8, edgecolors="none", label="Gen $k$ (aligned)")
for i in range(0, n, 8):
    ax.annotate("", xy=aligned[i], xytext=unaligned[i],
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.8))
ax.set_title("3. Temporal alignment\n($\\lambda=0.8$, Eq. 1)", fontsize=10, fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])
ax.spines[["top","right","left","bottom"]].set_visible(False)
ax.legend(fontsize=7, loc="upper left", framealpha=0.8)

# ── Panel 4: Velocity field ───────────────────────────────────────────────
ax = axes[3]
ax.set_facecolor(LIGHT)
x = np.linspace(-1.5, 1.5, 10)
y = np.linspace(-1.5, 1.5, 10)
X, Y = np.meshgrid(x, y)
cx, cy = 0.2, 0.1
U = cx - X
V = cy - Y
speed = np.sqrt(U**2 + V**2)
U /= speed + 0.3
V /= speed + 0.3
ax.streamplot(X, Y, U, V, color=speed, cmap="plasma", density=1.2,
              linewidth=0.9, arrowsize=0.9)
ax.scatter(*rng.normal([cx, cy], 0.3, (40, 2)).T, c="gold", s=20,
           edgecolors="none", alpha=0.8, zorder=5, label="high fitness")
ax.set_title("4. Velocity field\n(convergence basin)", fontsize=10, fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])
ax.spines[["top","right","left","bottom"]].set_visible(False)
ax.legend(fontsize=7, loc="upper right", framealpha=0.8)

fig.tight_layout()

PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
out = PAPER_FIGURES / "pipeline_diagram.pdf"
fig.savefig(out, bbox_inches="tight", facecolor="white")
out_png = PAPER_FIGURES / "pipeline_diagram.png"
fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved {out}")
print(f"Saved {out_png}")
