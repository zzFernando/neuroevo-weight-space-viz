"""Exp 23: graph embedding removes the permutation symmetry that weight space can't.

Demonstrates the claim from Exp 21/22: neuron-permuted copies of a network are the
SAME function but land far apart in weight space. We build several distinct parent
networks (make_moons MLP 2->16->1), generate permuted copies of each (functionally
identical), then compare two representations:
  - weight space      : flattened (48,) weight vector
  - graph spectrum     : sorted eigenvalues of the weighted graph Laplacian (perm-invariant)
A 2-D PCA of each shows permuted copies scattering in weight space but collapsing to a
single point per function in graph space. Quantified by within/between cluster ratio.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


from shared import FIGURES_DIR, RESULTS_DIR, ensure_dirs, set_science_style, save_results_csv

HIDDEN = 16
N_PARENTS = 4
N_PERM = 20
SEED = 42
rng = np.random.default_rng(SEED)


def split(flat):
    return flat[:2 * HIDDEN].reshape(2, HIDDEN), flat[2 * HIDDEN:].reshape(HIDDEN, 1)


def forward(flat, X):
    W1, W2 = split(flat)
    return 1.0 / (1.0 + np.exp(-(np.tanh(X @ W1) @ W2)[:, 0]))


def permute_hidden(flat, perm):
    """Permute the 16 hidden neurons — a functionally identical network."""
    W1, W2 = split(flat)
    return np.concatenate([W1[:, perm].ravel(), W2[perm, :].ravel()])


def graph_spectrum(flat):
    """Sorted eigenvalues of the weighted Laplacian of the MLP graph.

    Nodes: 2 inputs + 16 hidden + 1 output = 19. Edges weighted by |w|. The spectrum
    is invariant to relabeling (permuting) the hidden nodes.
    """
    W1, W2 = split(flat)
    n = 2 + HIDDEN + 1
    A = np.zeros((n, n))
    for i in range(2):                      # input i -> hidden j
        for j in range(HIDDEN):
            w = abs(W1[i, j]); A[i, 2 + j] = w; A[2 + j, i] = w
    for j in range(HIDDEN):                 # hidden j -> output
        w = abs(W2[j, 0]); A[2 + j, n - 1] = w; A[n - 1, 2 + j] = w
    L = np.diag(A.sum(1)) - A
    return np.sort(np.linalg.eigvalsh(L))   # (19,) sorted, perm-invariant


def cluster_ratio(emb, labels):
    """mean within-cluster distance / mean between-cluster distance."""
    within, between = [], []
    for i in range(len(emb)):
        for j in range(i + 1, len(emb)):
            d = np.linalg.norm(emb[i] - emb[j])
            (within if labels[i] == labels[j] else between).append(d)
    return float(np.mean(within)) / (float(np.mean(between)) + 1e-12)


def main():
    ensure_dirs()
    set_science_style()

    # distinct parent networks (random) + permuted copies of each
    parents = [rng.normal(0, 1.0, 2 * HIDDEN + HIDDEN) for _ in range(N_PARENTS)]
    W_vecs, spectra, labels = [], [], []
    X_check = rng.normal(0, 1, (50, 2))
    max_func_diff = 0.0
    for pid, p in enumerate(parents):
        base_out = forward(p, X_check)
        for _ in range(N_PERM):
            perm = rng.permutation(HIDDEN)
            child = permute_hidden(p, perm)
            max_func_diff = max(max_func_diff, np.abs(forward(child, X_check) - base_out).max())
            W_vecs.append(child)
            spectra.append(graph_spectrum(child))
            labels.append(pid)
    W_vecs = np.array(W_vecs); spectra = np.array(spectra); labels = np.array(labels)
    print(f"max functional difference among permuted copies: {max_func_diff:.2e}  (≈0 ⇒ identical fn)")

    emb_w = PCA(n_components=2, random_state=SEED).fit_transform(W_vecs)
    emb_g = PCA(n_components=2, random_state=SEED).fit_transform(spectra)

    r_w = cluster_ratio(W_vecs, labels)
    r_g = cluster_ratio(spectra, labels)
    print(f"within/between distance ratio  —  weight space: {r_w:.3f}   graph spectrum: {r_g:.3e}")
    print("(weight ratio ~1 ⇒ permutations as far as different functions; "
          "graph ratio ~0 ⇒ permutations collapse)")
    save_results_csv(
        [{"space": "weight", "within_between_ratio": round(r_w, 4)},
         {"space": "graph_spectrum", "within_between_ratio": float(f"{r_g:.3e}")}],
        RESULTS_DIR / "exp23_graph_embedding.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    cmap = plt.cm.tab10
    for ax, emb, title, ratio in [
        (axes[0], emb_w, "Weight space (48-dim)", r_w),
        (axes[1], emb_g, "Graph spectrum (Laplacian eigenvalues)", r_g),
    ]:
        ax.set_facecolor("white")
        for pid in range(N_PARENTS):
            m = labels == pid
            ax.scatter(emb[m, 0], emb[m, 1], s=30, alpha=0.8, color=cmap(pid),
                       label=f"function {pid + 1}", edgecolors="k", linewidths=0.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title}\nwithin/between dist ratio = {ratio:.2e}", fontsize=10)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Permuted (functionally identical) networks: weight space scatters them, "
                 "graph spectrum collapses them", fontsize=11, y=1.02)
    out = FIGURES_DIR / "exp23_graph_embedding_demo.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
