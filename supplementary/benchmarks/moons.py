from __future__ import annotations

import random
from typing import List, Sequence

import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from .base import NeuroEvoBase


class NeuroEvoMoons(NeuroEvoBase):
    """Neuroevolution on make_moons binary classification (2D input)."""

    def __init__(
        self,
        pop_size: int = 50,
        hidden_dim: int = 16,
        mutation_rate: float = 0.05,
        seed: int = 42,
    ) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.weight_init_mean = 0.0
        self.weight_init_std = 0.5

        X, y = make_moons(n_samples=1000, noise=0.25, random_state=seed)
        scaler = StandardScaler().fit(X)
        self.X = scaler.transform(X)
        self.y = y.reshape(-1, 1).astype(float)

        self.shapes = [(2, hidden_dim), (hidden_dim, 1)]
        self.population = [self.random_individual() for _ in range(pop_size)]

    def forward(self, individual: Sequence[np.ndarray], X: np.ndarray) -> np.ndarray:
        w1, w2 = individual
        h = np.tanh(X @ w1)
        o = h @ w2
        return 1.0 / (1.0 + np.exp(-o))

    def evaluate(self, individual: Sequence[np.ndarray]) -> float:
        preds = self.forward(individual, self.X)
        eps = 1e-8
        loss = -(self.y * np.log(preds + eps) + (1 - self.y) * np.log(1 - preds + eps)).mean()
        return -float(loss)
