from __future__ import annotations

import os
import pickle
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import NeuroEvoBase

_CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_CACHE_DIR = Path.home() / ".cache" / "neuroevo_viz"


def _load_cifar10_raw() -> tuple[np.ndarray, np.ndarray]:
    """Download (once) and load CIFAR-10 train batch 1. Returns X (N,3072), y (N,)."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = _CACHE_DIR / "cifar-10-python.tar.gz"
    batch_path = _CACHE_DIR / "cifar-10-batches-py" / "data_batch_1"

    if not batch_path.exists():
        urllib.request.urlretrieve(_CIFAR10_URL, tar_path)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(_CACHE_DIR)

    with open(batch_path, "rb") as f:
        d = pickle.load(f, encoding="bytes")

    X = d[b"data"].astype(np.float32)
    y = np.array(d[b"labels"], dtype=np.int64)
    return X, y


class NeuroEvoCIFAR10(NeuroEvoBase):
    """
    Neuroevolution on CIFAR-10: 10-class classification.
    Input: 3072D (32x32x3). Subsamples to n_samples for speed.
    Uses PCA pre-reduction for UMAP (handled externally via pca_dims attribute).
    """

    pca_dims: int = 50  # consumed by compute_aligned_umap_embedding

    def __init__(
        self,
        pop_size: int = 20,
        hidden_dim: int = 64,
        mutation_rate: float = 0.05,
        seed: int = 42,
        n_samples: int = 3000,
    ) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.weight_init_mean = 0.0
        self.weight_init_std = 0.1  # smaller init for high-dim input

        X_raw, y_raw = _load_cifar10_raw()
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_raw), size=min(n_samples, len(X_raw)), replace=False)
        X_sub, y_sub = X_raw[idx], y_raw[idx]

        scaler = StandardScaler().fit(X_sub)
        self.X = scaler.transform(X_sub)
        self.y = y_sub

        self.n_classes = 10
        self.shapes = [(3072, hidden_dim), (hidden_dim, self.n_classes)]
        self.population = [self.random_individual() for _ in range(pop_size)]

    def forward(self, individual: Sequence[np.ndarray], X: np.ndarray) -> np.ndarray:
        w1, w2 = individual
        h = np.tanh(X @ w1)
        logits = h @ w2
        # numerically stable softmax
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_l = np.exp(shifted)
        return exp_l / exp_l.sum(axis=1, keepdims=True)

    def evaluate(self, individual: Sequence[np.ndarray]) -> float:
        probs = self.forward(individual, self.X)
        eps = 1e-8
        log_p = np.log(probs[np.arange(len(self.y)), self.y] + eps)
        return float(log_p.mean())  # negative cross-entropy (maximize)
