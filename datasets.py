"""
Dataset implementations for neuroevolution weight-space visualization.

Each class shares the same interface:
  - __init__(pop_size, hidden_dim, mutation_rate, seed)
  - population: List of individuals
  - flatten(individual) -> np.ndarray
  - evaluate(individual) -> float
  - evolve_one_generation(elite_frac, min_elite) -> (best_idx, best_fit)

Add new datasets by subclassing NeuroEvoBase and registering in DATASETS.
"""

from __future__ import annotations

import random
from typing import List, Sequence

import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class NeuroEvoBase:
    """Shared neuroevolution mechanics. Subclasses supply __init__ and evaluate()."""

    pop_size: int
    mutation_rate: float
    shapes: list
    rng: np.random.Generator
    population: list

    def flatten(self, individual: Sequence[np.ndarray]) -> np.ndarray:
        return np.concatenate([w.ravel() for w in individual])

    def random_individual(self) -> List[np.ndarray]:
        return [self.rng.normal(0.0, 0.5, size=s) for s in self.shapes]

    def mutate(self, individual: Sequence[np.ndarray]) -> List[np.ndarray]:
        return [w + self.rng.normal(0, self.mutation_rate, size=w.shape) for w in individual]

    def evolve_one_generation(self, elite_frac: float = 0.2, min_elite: int = 2):
        fitness = np.array([self.evaluate(ind) for ind in self.population])
        n_elite = max(min_elite, int(self.pop_size * elite_frac))
        elite_idx = np.argsort(fitness)[-n_elite:]
        elites = [self.population[i] for i in elite_idx]
        new_pop: list = elites.copy()
        while len(new_pop) < self.pop_size:
            parent = elites[self.rng.integers(0, len(elites))]
            new_pop.append(self.mutate(parent))
        self.population = new_pop
        new_fitness = np.array([self.evaluate(ind) for ind in self.population])
        best_idx = int(np.argmax(new_fitness))
        return best_idx, float(new_fitness[best_idx])


# ---------------------------------------------------------------------------
# Make Moons
# ---------------------------------------------------------------------------

class NeuroEvoMoons(NeuroEvoBase):
    """
    Binary classification on make_moons.
    Architecture: 2 → hidden_dim → 1 (sigmoid).
    """

    def __init__(self, pop_size: int = 50, hidden_dim: int = 16,
                 mutation_rate: float = 0.05, seed: int = 42) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.shapes = [(2, hidden_dim), (hidden_dim, 1)]

        X, y = make_moons(n_samples=1000, noise=0.25, random_state=seed)
        self.X = StandardScaler().fit_transform(X)
        self.y = y.reshape(-1, 1).astype(np.float32)
        self.population = [self.random_individual() for _ in range(pop_size)]

    def _forward(self, individual, X: np.ndarray) -> np.ndarray:
        w1, w2 = individual
        h = np.tanh(X @ w1)
        return 1.0 / (1.0 + np.exp(-(h @ w2)))

    def evaluate(self, individual) -> float:
        p = self._forward(individual, self.X)
        eps = 1e-8
        loss = -(self.y * np.log(p + eps) + (1 - self.y) * np.log(1 - p + eps)).mean()
        return -float(loss)


# ---------------------------------------------------------------------------
# MNIST — binary (digit 0 vs. rest)
# ---------------------------------------------------------------------------

class NeuroEvoMNIST(NeuroEvoBase):
    """
    Binary digit classification: digit-0 vs. rest.

    Uses sklearn's built-in load_digits() (8×8 pixels, 64-D) — no download needed.
    Architecture: 64 → hidden_dim → 1 (sigmoid).
    """

    def __init__(self, pop_size: int = 30, hidden_dim: int = 32,
                 mutation_rate: float = 0.05, seed: int = 42) -> None:
        from sklearn.datasets import load_digits

        random.seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.shapes = [(64, hidden_dim), (hidden_dim, 1)]

        digits = load_digits()
        X, y = digits.data.astype(np.float32), digits.target
        self.X = StandardScaler().fit_transform(X)
        self.y = (y == 0).astype(np.float32).reshape(-1, 1)
        self.population = [self.random_individual() for _ in range(pop_size)]

    def _forward(self, individual, X: np.ndarray) -> np.ndarray:
        w1, w2 = individual
        h = np.tanh(X @ w1)
        return 1.0 / (1.0 + np.exp(-(h @ w2)))

    def evaluate(self, individual) -> float:
        p = self._forward(individual, self.X)
        eps = 1e-8
        loss = -(self.y * np.log(p + eps) + (1 - self.y) * np.log(1 - p + eps)).mean()
        return -float(loss)


# ---------------------------------------------------------------------------
# BRAX Ant — simplified locomotion proxy (numpy only)
# ---------------------------------------------------------------------------

class NeuroEvoAnt(NeuroEvoBase):
    """
    Simplified locomotion task — numpy proxy for BRAX Ant.

    Observation: 27-D  |  Actions: 8-D  |  Episode: 100 steps.
    A single-body forward locomotion model with 8 joints. Fitness is cumulative
    forward velocity minus a small control cost — enough to produce realistic
    weight-space dynamics without requiring JAX or MuJoCo.
    Architecture: 27 → hidden_dim → 8 (tanh).
    """

    OBS_DIM = 27
    ACT_DIM = 8
    N_STEPS = 100
    DT = 0.05

    def __init__(self, pop_size: int = 30, hidden_dim: int = 64,
                 mutation_rate: float = 0.05, seed: int = 42) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.shapes = [(self.OBS_DIM, hidden_dim), (hidden_dim, self.ACT_DIM)]
        self.population = [self.random_individual() for _ in range(pop_size)]

    def _obs(self, x: float, x_dot: float,
             joints: np.ndarray, joint_vels: np.ndarray) -> np.ndarray:
        raw = np.concatenate([
            [x, x_dot],
            np.sin(joints),
            np.cos(joints),
            joint_vels,
            [0.0, 0.0, 0.0],  # z, roll, pitch (held at 0 in 2-D proxy)
        ])
        return raw[: self.OBS_DIM]

    def _forward(self, individual, obs: np.ndarray) -> np.ndarray:
        w1, w2 = individual
        return np.tanh(np.tanh(obs @ w1) @ w2)

    def _simulate(self, individual) -> float:
        x, x_dot = 0.0, 0.0
        joints = np.zeros(self.ACT_DIM)
        joint_vels = np.zeros(self.ACT_DIM)
        total = 0.0
        for _ in range(self.N_STEPS):
            obs = self._obs(x, x_dot, joints, joint_vels)
            action = self._forward(individual, obs)
            joint_vels = 0.9 * joint_vels + 0.1 * action
            joints = joints + self.DT * joint_vels
            x_ddot = 0.3 * float(np.dot(action, np.cos(joints))) - 0.1 * x_dot
            x_dot += self.DT * x_ddot
            x += self.DT * x_dot
            total += max(0.0, x_dot) - 0.001 * float(np.dot(action, action))
        return total

    def evaluate(self, individual) -> float:
        return self._simulate(individual)


# ---------------------------------------------------------------------------
# ImageNet proxy — CIFAR-10 (10-class, 3 072-D input)
# ---------------------------------------------------------------------------

def _load_cifar10(n_samples: int = 3000, seed: int = 42):
    """
    Download CIFAR-10 binary directly from the University of Toronto.
    Cache in ~/.cache/neuroevo/cifar-10-python.tar.gz.
    Returns (X, y) with X in [0,255] float32, shape (n_samples, 3072).
    """
    import io
    import pickle
    import tarfile
    import urllib.request
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "neuroevo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tar_path = cache_dir / "cifar-10-python.tar.gz"

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    if not tar_path.exists():
        urllib.request.urlretrieve(url, tar_path)

    Xs, ys = [], []
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            if "data_batch" in member.name:
                f = tf.extractfile(member)
                batch = pickle.load(io.BytesIO(f.read()), encoding="bytes")
                Xs.append(batch[b"data"])
                ys.append(np.array(batch[b"labels"]))

    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(int)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    return X[idx], y[idx]


class NeuroEvoCIFAR10(NeuroEvoBase):
    """
    10-class classification on CIFAR-10, used as an ImageNet-scale proxy.

    Downloads directly from cs.toronto.edu on first use (~170 MB, cached in
    ~/.cache/neuroevo/). Subsamples 3 000 points for speed.
    Architecture: 3072 → hidden_dim → 10 (softmax).
    """

    def __init__(self, pop_size: int = 20, hidden_dim: int = 64,
                 mutation_rate: float = 0.05, seed: int = 42) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.shapes = [(3072, hidden_dim), (hidden_dim, 10)]

        X, y = _load_cifar10(n_samples=3000, seed=seed)
        self.X = StandardScaler().fit_transform(X)
        self.y = y
        self.population = [self.random_individual() for _ in range(pop_size)]

    def _forward(self, individual, X: np.ndarray) -> np.ndarray:
        w1, w2 = individual
        logits = np.tanh(X @ w1) @ w2
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def evaluate(self, individual) -> float:
        p = self._forward(individual, self.X)
        eps = 1e-8
        return float(np.log(p[np.arange(len(self.y)), self.y] + eps).mean())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "make_moons": {
        "class": NeuroEvoMoons,
        "label": {"pt": "Make Moons", "en": "Make Moons"},
        "desc": {
            "pt": "Classificação binária 2D — tarefa clássica de neuroevolução.",
            "en": "2D binary classification — classic neuroevolution benchmark.",
        },
        "default_hidden": 16,
        "default_pop": 50,
        "requires_download": False,
    },
    "mnist": {
        "class": NeuroEvoMNIST,
        "label": {"pt": "MNIST", "en": "MNIST"},
        "desc": {
            "pt": "Dígitos 8×8 (64-D) — sklearn built-in, sem download. Binário: dígito 0 vs. resto.",
            "en": "8×8 digits (64-D) — sklearn built-in, no download. Binary: digit 0 vs. rest.",
        },
        "default_hidden": 32,
        "default_pop": 30,
        "requires_download": False,
    },
    "brax_ant": {
        "class": NeuroEvoAnt,
        "label": {"pt": "BRAX Ant (proxy)", "en": "BRAX Ant (proxy)"},
        "desc": {
            "pt": "Locomoção simplificada (proxy numpy do BRAX Ant). Obs: 27-D, Ações: 8-D.",
            "en": "Simplified locomotion task (numpy proxy for BRAX Ant). Obs: 27-D, Actions: 8-D.",
        },
        "default_hidden": 64,
        "default_pop": 30,
        "requires_download": False,
    },
    "imagenet": {
        "class": NeuroEvoCIFAR10,
        "label": {"pt": "ImageNet (CIFAR-10)", "en": "ImageNet (CIFAR-10)"},
        "desc": {
            "pt": "CIFAR-10 como proxy ImageNet-scale (3072-D, 10 classes). ⚠️ Download na 1ª execução (~170 MB, cs.toronto.edu).",
            "en": "CIFAR-10 as ImageNet-scale proxy (3072-D, 10 classes). ⚠️ Download on first run (~170 MB, cs.toronto.edu).",
        },
        "default_hidden": 64,
        "default_pop": 20,
        "requires_download": True,
    },
}


def build_dataset(name: str, pop_size: int, hidden_dim: int,
                  mutation_rate: float, seed: int) -> NeuroEvoBase:
    """Instantiate a NeuroEvo problem by dataset name."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS)}")
    cls = DATASETS[name]["class"]
    return cls(pop_size=pop_size, hidden_dim=hidden_dim,
               mutation_rate=mutation_rate, seed=seed)
