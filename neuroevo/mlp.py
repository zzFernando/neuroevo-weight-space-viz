"""Fixed-topology Multi-Layer Perceptron for neuroevolution."""
import numpy as np


class FixedMLP:
    """A fixed-topology MLP with configurable layers."""
    
    def __init__(self, layer_sizes):
        """
        Initialize MLP with fixed architecture.
        
        Args:
            layer_sizes: List of integers specifying the size of each layer
                        (e.g., [4, 10, 3] for input=4, hidden=10, output=3)
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        
        # Calculate total number of weights (including biases)
        self.weight_shapes = []
        self.total_weights = 0
        
        for i in range(self.num_layers):
            # Weights: layer_sizes[i+1] x layer_sizes[i]
            # Biases: layer_sizes[i+1]
            weight_shape = (layer_sizes[i+1], layer_sizes[i])
            bias_shape = (layer_sizes[i+1],)
            self.weight_shapes.append((weight_shape, bias_shape))
            self.total_weights += np.prod(weight_shape) + np.prod(bias_shape)
    
    def set_weights(self, flat_weights):
        """
        Set network weights from a flattened array.
        
        Args:
            flat_weights: 1D numpy array of all weights and biases
        """
        if len(flat_weights) != self.total_weights:
            raise ValueError(f"Expected {self.total_weights} weights, got {len(flat_weights)}")
        
        self.weights = []
        self.biases = []
        idx = 0
        
        for weight_shape, bias_shape in self.weight_shapes:
            # Extract weights for this layer
            weight_size = np.prod(weight_shape)
            w = flat_weights[idx:idx+weight_size].reshape(weight_shape)
            idx += weight_size
            
            # Extract biases for this layer
            bias_size = np.prod(bias_shape)
            b = flat_weights[idx:idx+bias_size].reshape(bias_shape)
            idx += bias_size
            
            self.weights.append(w)
            self.biases.append(b)
    
    def get_weights(self):
        """
        Get flattened array of all weights and biases.
        
        Returns:
            1D numpy array of all weights and biases
            
        Raises:
            AttributeError: If set_weights has not been called first
        """
        if not hasattr(self, 'weights') or not hasattr(self, 'biases'):
            raise AttributeError("Weights not initialized. Call set_weights() first.")
        
        flat = []
        for w, b in zip(self.weights, self.biases):
            flat.append(w.flatten())
            flat.append(b.flatten())
        return np.concatenate(flat)
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Args:
            X: Input data, shape (n_samples, input_size) or (input_size,)
            
        Returns:
            Output activations, shape (n_samples, output_size) or (output_size,)
            
        Raises:
            AttributeError: If set_weights has not been called first
        """
        if not hasattr(self, 'weights') or not hasattr(self, 'biases'):
            raise AttributeError("Weights not initialized. Call set_weights() first.")
        
        # Handle single sample
        single_sample = False
        if X.ndim == 1:
            X = X.reshape(1, -1)
            single_sample = True
        
        activation = X
        
        # Forward through all layers
        for i in range(self.num_layers):
            z = activation @ self.weights[i].T + self.biases[i]
            
            # Apply activation function (ReLU for hidden, no activation for output)
            if i < self.num_layers - 1:
                activation = np.maximum(0, z)  # ReLU
            else:
                activation = z  # No activation on output layer
        
        if single_sample:
            return activation.flatten()
        return activation
    
    def predict(self, X):
        """
        Make predictions (class labels for classification).
        
        Args:
            X: Input data, shape (n_samples, input_size)
            
        Returns:
            Predicted class labels, shape (n_samples,)
        """
        output = self.forward(X)
        return np.argmax(output, axis=1)
