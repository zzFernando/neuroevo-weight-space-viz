"""Exp 29: signed graph embedding — fixing the |w|-spectrum's blindness to sign.

Exp 23 showed the Laplacian spectrum of the |w| graph is permutation-invariant, but
flagged a limitation: using |w| discards the sign of the weights, so two networks with
identical |w| but different signs (hence different functions) get the SAME spectrum.
Here we close that: we compare three representations on a sharp test —
  - weight vector            (baseline)
  - |w| Laplacian spectrum   (Exp 23)
  - signed Laplacian spectrum (this experiment)
against two transformations of a base net:
  - permutation of hidden neurons  → SAME function (should collapse: distance ≈ 0)
  - sign flip of a random weight subset → DIFFERENT function, identical |w|
    (should separate: distance > 0)

Good representation: collapses permutations AND separates the sign-changed (different)
function. The |w| spectrum collapses both (blind to sign); the signed spectrum fixes it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.shared import FIGURES_DIR, RESULTS_DIR, ensure_dirs, set_science_style, save_results_csv

HIDDEN = 16
N_BASE = 6        # distinct base networks
N_VARIANTS = 10   # copies per transformation
SEED = 42
rng = np.random.default_rng(SEED)


def split(flat):
    return flat[:2 * HIDDEN].reshape(2, HIDDEN), flat[2 * HIDDEN:].reshape(HIDDEN, 1)


def forward(flat, X):
    W1, W2 = split(flat)
    return 1.0 / (1.0 + np.exp(-(np.tanh(X @ W1) @ W2)[:, 0]))


def permute_hidden(flat, perm):
    W1, W2 = split(flat)
    return np.concatenate([W1[:, perm].ravel(), W2[perm, :].ravel()])


def sign_flip(flat, frac=0.3):
    """Flip the sign of a random subset of weights — identical |w|, different function."""
    f = flat.copy()
    idx = rng.choice(len(f), size=int(frac * len(f)), replace=False)
    f[idx] *= -1
    return f


def laplacian_spectrum(flat, signed: bool):
    """Sorted eigenvalues of the (signed or |w|) graph Laplacian of the MLP."""
    W1, W2 = split(flat)
    n = 2 + HIDDEN + 1
    A = np.zeros((n, n))
    for i in range(2):
        for j in range(HIDDEN):
            w = W1[i, j] if signed else abs(W1[i, j])
            A[i, 2 + j] = w; A[2 + j, i] = w
    for j in range(HIDDEN):
        w = W2[j, 0] if signed else abs(W2[j, 0])
        A[2 + j, n - 1] = w; A[n - 1, 2 + j] = w
    D = np.diag(np.abs(A).sum(1))      # signed Laplacian uses |·| row sums (Kunegis et al.)
    return np.sort(np.linalg.eigvalsh(D - A))


def main():
    ensure_dirs()
    set_science_style()

    X = rng.normal(0, 1, (60, 2))
    reps = {"weight vector": lambda f: f,
            "|w| spectrum (Exp 23)": lambda f: laplacian_spectrum(f, signed=False),
            "signed spectrum (Exp 29)": lambda f: laplacian_spectrum(f, signed=True)}

    # accumulate normalized distances per representation
    func_perm, func_sign = [], []
    dist = {r: {"perm": [], "sign": []} for r in reps}

    for _ in range(N_BASE):
        base = rng.normal(0, 1.0, 2 * HIDDEN + HIDDEN)
        base_out = forward(base, X)
        base_vecs = {r: fn(base) for r, fn in reps.items()}
        # scale per representation = typical distance between two random nets (for normalization)
        rand_ref = rng.normal(0, 1.0, 2 * HIDDEN + HIDDEN)
        scale = {r: np.linalg.norm(reps[r](rand_ref) - base_vecs[r]) + 1e-9 for r in reps}

        for _ in range(N_VARIANTS):
            p = permute_hidden(base, rng.permutation(HIDDEN))
            s = sign_flip(base)
            func_perm.append(float(np.abs(forward(p, X) - base_out).max()))
            func_sign.append(float(np.abs(forward(s, X) - base_out).max()))
            for r, fn in reps.items():
                dist[r]["perm"].append(np.linalg.norm(fn(p) - base_vecs[r]) / scale[r])
                dist[r]["sign"].append(np.linalg.norm(fn(s) - base_vecs[r]) / scale[r])

    print(f"functional diff — permutation: {np.mean(func_perm):.2e} (≈0, same fn) | "
          f"sign-flip: {np.mean(func_sign):.3f} (>0, different fn)")
    rows = []
    print(f"\n{'representation':26s} {'d(perm)':>10s} {'d(sign)':>10s}  veredito")
    for r in reps:
        dp, ds = np.mean(dist[r]["perm"]), np.mean(dist[r]["sign"])
        if dp < 0.05 and ds > 0.05:
            verdict = "invariante a permutação E sensível a sinal ✓"
        elif dp < 0.05 and ds <= 0.05:
            verdict = "colapsa permutação MAS cego a sinal ✗"
        else:
            verdict = "não invariante a permutação (sensível a tudo) ✗"
        rows.append({"representation": r, "dist_permutation": round(float(dp), 4),
                     "dist_signflip": round(float(ds), 4), "verdict": verdict})
        print(f"  {r:24s} {dp:10.4f} {ds:10.4f}  {verdict}")
    save_results_csv(rows, RESULTS_DIR / "exp29_signed_graph.csv")

    # figure: grouped bars, normalized distance for perm vs sign-flip per representation
    fig, ax = plt.subplots(figsize=(8.5, 5))
    labels = list(reps)
    x = np.arange(len(labels)); w = 0.36
    dperm = [np.mean(dist[r]["perm"]) for r in labels]
    dsign = [np.mean(dist[r]["sign"]) for r in labels]
    ax.bar(x - w/2, dperm, w, label="permutação (mesma função)", color="#1f77b4")
    ax.bar(x + w/2, dsign, w, label="sign-flip (função diferente)", color="#d62728")
    ax.axhline(0.05, ls=":", color="#666", lw=1)
    ax.annotate("≈0 desejado p/ mesma função", (len(labels)-1, 0.06), fontsize=8, color="#666")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("distância normalizada ao original")
    ax.set_title("Distância normalizada: o espectro assinado separa funções diferentes\n"
                 "(sign-flip) mantendo invariância à permutação — o |w| é cego ao sinal", fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "exp29_signed_graph_embedding.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
