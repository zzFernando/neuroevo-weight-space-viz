"""Fixed-topology Multi-Layer Perceptron for neuroevolution.

This module provides a pure-Numpy MLP implementation with a richer default
hidden architecture (used when only input/output sizes are provided), ELU
activation for hidden layers, and an optional deterministic dropout parameter.
"""
import numpy as np


class FixedMLP:
    """A fixed-topology MLP with configurable layers.

    If `layer_sizes` has length 2 (input, output) the MLP expands to
    `[input, 128, 64, 32, output]` to provide higher capacity by default.
    """

    def __init__(self, layer_sizes, dropout: float = 0.0):
        # If user provided only input/output, expand to a richer hidden architecture
        if len(layer_sizes) == 2:
            in_sz, out_sz = layer_sizes
            hidden = [128, 64, 32]
            layer_sizes = [in_sz] + hidden + [out_sz]

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.dropout = float(dropout)

        # Calculate total number of weights (including biases)
        self.weight_shapes = []
        self.total_weights = 0

        for i in range(self.num_layers):
            weight_shape = (layer_sizes[i + 1], layer_sizes[i])
            bias_shape = (layer_sizes[i + 1],)
            self.weight_shapes.append((weight_shape, bias_shape))
            self.total_weights += int(np.prod(weight_shape) + np.prod(bias_shape))

    def set_weights(self, flat_weights):
        if len(flat_weights) != self.total_weights:
            raise ValueError(f"Expected {self.total_weights} weights, got {len(flat_weights)}")

        self.weights = []
        self.biases = []
        idx = 0

        for weight_shape, bias_shape in self.weight_shapes:
            weight_size = int(np.prod(weight_shape))
            w = flat_weights[idx:idx + weight_size].reshape(weight_shape)
            idx += weight_size

            bias_size = int(np.prod(bias_shape))
            b = flat_weights[idx:idx + bias_size].reshape(bias_shape)
            idx += bias_size

            self.weights.append(w)
            self.biases.append(b)

    def get_weights(self):
        if not hasattr(self, 'weights') or not hasattr(self, 'biases'):
            raise AttributeError("Weights not initialized. Call set_weights() first.")

        flat = []
        for w, b in zip(self.weights, self.biases):
            flat.append(w.flatten())
            flat.append(b.flatten())
        return np.concatenate(flat)

    def forward(self, X):
        if not hasattr(self, 'weights') or not hasattr(self, 'biases'):
            raise AttributeError("Weights not initialized. Call set_weights() first.")

        single_sample = False
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            single_sample = True

        activation = X

        for i in range(self.num_layers):
            z = activation @ self.weights[i].T + self.biases[i]

            # Hidden: ELU activation
            if i < self.num_layers - 1:
                activation = np.where(z > 0, z, np.expm1(z))
                if self.dropout and self.dropout > 0.0:
                    # deterministic mask based on layer index and feature index
                    mask = (np.arange(activation.shape[1]) % max(1, int(1 / max(1e-6, self.dropout)))) != 0
                    activation = activation * mask
            else:
                activation = z

        if single_sample:
            return activation.flatten()
        return activation

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
