import numpy as np


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape.

        :return out: Outputs, of the same shape as x.
        :return cache: Cache, stored for backward computation, of the same shape as x.
        """
        out, cache = 1 / (1 + np.exp(-x)), x
        return out, cache

    def backward(self, dout, cache):
        """
        :param dout: Upstream gradient from the computational graph, from the Loss function
                    and up to this layer. Has the shape of the output of forward().
        :param cache: The values that were stored during forward() to the memory,
                    to be used during the backpropogation.
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        x = cache
        dx = dout * (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
        return dx


class Relu:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape.

        :return outputs: Outputs, of the same shape as x.
        :return cache: Cache, stored for backward computation, of the same shape as x.
        """
        out = np.where(x > 0, x, 0)
        cache = x
        return out, cache

    def backward(self, dout, cache):
        """
        :param dout: Upstream gradient from the computational graph, from the Loss function
                    and up to this layer. Has the shape of the output of forward().
        :param cache: The values that were stored during forward() to the memory,
                    to be used during the backpropogation.
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        x = cache
        dx = dout * np.where(x > 0, 1, 0)
        return dx
    
class Tanh:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape

        :return out: Output, of the same shape as x
        :return cache: Cache, for backward computation, of the same shape as x
        """
        outputs = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
        cache = x
        return outputs, cache

    def backward(self, dout, cache):
        """
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        x = cache
        dx = (1 - ((np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x)))**2)*dout
        return dx

class Affine:
    """
    An affine (fully connected) layer.
    This layer computes out = x @ w + b.
    During the backward pass, the gradients with respect to w and b are stored.
    """
    def __init__(self, input_shape=784, number_of_units=32, bias=None):
        # Xavier normal initialization for weights.
        fan_in = input_shape
        fan_out = number_of_units
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.w = np.random.randn(input_shape, number_of_units) * std
        # Initialize biases
        if bias is not None:
            self.b = np.full((number_of_units,), bias)  # Initialize with provided value
        else:
            self.b = np.zeros((number_of_units,))  # Default to zeros

        # Gradients (populated during backward)
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        :param x: Input data of shape (N, d1, ..., d_k).
        :return out: Output of shape (N, M).
        :return cache: Tuple (x, w, b) for use in backward.
        """
        self.x_shape = x.shape  # Store the original shape for later reshaping.
        N = x.shape[0]
        # Flatten the input: shape becomes (N, D)
        self.x_flat = x.reshape(N, -1)
        out = self.x_flat @ self.w + self.b
        cache = (x, self.w, self.b)
        return out, cache

    def backward(self, dout, cache):
        """
        :param dout: Upstream derivative, of shape (N, M).
        :param cache: Tuple (x, w, b) from the forward pass.
        :return dx: Gradient with respect to x, with the same shape as x.
        """
        x, w, b = cache
        N = x.shape[0]
        x_flat = x.reshape(N, -1)
        # Gradient with respect to x
        dx = (dout @ w.T).reshape(x.shape)
        # Store gradients with respect to w and b in the layer.
        self.dw = x_flat.T @ dout
        self.db = np.sum(dout, axis=0)
        return dx



class Model:
    def __init__(self, layers=None):
        # Initialize layers as an empty list if no argument is provided
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

    def add_layer(self, layer):
        """Add a layer to the model."""
        self.layers.append(layer)

    def forward(self, x):
        """
        Pass the input through all layers.
        Caches are stored from each layer for use during backward pass.
        """
        self.caches = []  # Reset caches at the start of each forward pass
        for layer in self.layers:
            x, cache = layer.forward(x)
            self.caches.append(cache)
        return x

    def backward(self, dout):
        """
        Perform backpropagation through the network by calling
        each layer's backward method in reverse order.
        """
        # Loop through layers in reverse order along with their caches.
        for layer, cache in zip(reversed(self.layers), reversed(self.caches)):
            # For layers like Affine that return multiple gradients,
            # you might want to store or update parameters here.
            dout = layer.backward(dout, cache)
        return dout

    def parameters(self):
        """
        Gather parameters from all layers that have trainable parameters.
        This returns a dictionary where keys might be names like 'layer0_w', 'layer0_b', etc.
        """
        params = {}
        for i, layer in enumerate(self.layers):
            # Check if the layer has weights and biases (as in the Affine layer)
            if hasattr(layer, 'w') and hasattr(layer, 'b'):
                params[f'layer{i}_w'] = layer.w
                params[f'layer{i}_b'] = layer.b
        return params

    def __call__(self, x):
        """Allow the model to be called like a function, e.g., output = model(x)"""
        return self.forward(x)