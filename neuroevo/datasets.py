"""Single complex dataset generator used throughout the experiments."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles


def generate_complex_dataset(n_samples: int = 6000,
                             n_classes: int = 4,
                             seed: int = 0):
    """Create a multi-manifold dataset with rich nonlinear structure.

    The generator mixes spirals, moons, circles and noisy blobs, projects them
    into a higher-dimensional space, adds sinusoidal distortions and injects
    label noise. The goal is to build a landscape where weight evolution really
    matters.
    """
    rng = np.random.default_rng(seed)
    parts = []
    labels = []
    per_component = max(50, n_samples // (n_classes * 2))

    for comp in range(n_classes * 2):
        size = per_component

        if comp % 4 == 0:
            theta = np.sqrt(rng.random(size)) * 3 * np.pi
            radius = 1.5 * theta + rng.normal(scale=0.3, size=size)
            part = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
        elif comp % 4 == 1:
            x, y = make_moons(n_samples=size, noise=0.25, random_state=rng.integers(0, 10000))
            part = np.column_stack((x, y))
        elif comp % 4 == 2:
            x, y = make_circles(n_samples=size, noise=0.2, factor=0.35,
                                random_state=rng.integers(0, 10000))
            part = np.column_stack((x, y))
        else:
            center = rng.normal(scale=2.5, size=2)
            part = center + rng.normal(scale=0.7, size=(size, 2))

        part = np.column_stack([
            part[:, 0],
            part[:, 1],
            np.sin(part[:, 0] * 2.0) + 0.2 * rng.normal(size=size)
        ])

        random_proj = rng.normal(scale=0.6, size=(3, 6))
        projected = np.tanh(part @ random_proj)
        higher = np.hstack([part, projected])

        parts.append(higher)
        labels.append(np.full(size, comp % n_classes, dtype=int))

    X = np.vstack(parts)
    y = np.hstack(labels)

    idx = rng.permutation(len(X))[:n_samples]
    X = X[idx]
    y = y[idx]

    flips = int(0.08 * len(y))
    if flips > 0:
        flip_idx = rng.choice(len(y), size=flips, replace=False)
        y[flip_idx] = rng.integers(0, n_classes, size=len(flip_idx))

    extra_noise = np.sin(X[:, :3]) + 0.3 * rng.normal(size=(len(X), 3))
    X = np.hstack([X, extra_noise])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train.astype(np.float32), X_test.astype(np.float32), y_train.astype(int), y_test.astype(int)
