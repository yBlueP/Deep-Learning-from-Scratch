import numpy as np

import common.functions as F


# ReLU activation layer
class ReLU:
    # Initialize with mask for negative values
    def __init__(self):
        self.mask = None

    # Forward pass: Apply ReLU (set negatives to 0)
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    # Backward pass: Zero gradients for negative inputs
    def backward(self, dout):
        dout[self.mask] = 0
        return dout
    
    # Callable interface
    def __call__(self, *args: np.array, **kwds) -> np.array:
        return self.forward(*args, **kwds)


# Sigmoid activation layer
class Sigmoid:
    # Initialize
    def __init__(self):
        pass

    # Forward pass: Compute sigmoid
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    # Backward pass: Compute gradient
    def backward(self, dout):
        return dout * (1 - self.out) * self.out

    # Callable interface
    def __call__(self, *args: np.array, **kwds) -> np.array:
        return self.forward(*args, **kwds)


# Softmax with Cross-Entropy Loss layer
class SoftmaxWithLoss:
    # Initialize
    def __init__(self):
        self.y = None
        self.t = None

    # Forward pass: Compute softmax and loss
    def forward(self, x, t):
        self.t = t
        self.y = F.softmax(x)
        return F.cross_entropy_error(self.y, self.t)

    # Backward pass: Compute gradient
    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        return dout * (self.y - self.t) / batch_size
    
    # Callable interface
    def __call__(self, *args: np.array, **kwds) -> np.array:
        return self.forward(*args, **kwds)


# Linear (Fully Connected) Layer
class LinearLayer:
    # Initialize with weights, biases, and activation
    def __init__(self, in_size, out_size, activation = ReLU(), weight_init_std = 0.01):
        self.W = np.random.randn(in_size, out_size) * weight_init_std
        self.b = np.random.randn(out_size) * weight_init_std
        self.x = None
        self.activation = activation
        self.gradient_w = None
        self.gradient_b = None

    # Forward pass: Linear transformation + activation
    def forward(self, x):
        self.x = x
        return self.activation(np.dot(x, self.W) + self.b)

    # Set weights
    def init_weight(self, w):
        self.W = w

    # Set biases
    def init_bias(self, b):
        self.b = b
    
    # Backward pass: Compute gradients and propagate error
    def backward(self, dout):
        dout = self.activation.backward(dout)
        self.gradient_w = np.dot(self.x.T, dout)
        self.gradient_b = np.sum(dout, axis=0)
        return np.dot(dout, self.W.T)


