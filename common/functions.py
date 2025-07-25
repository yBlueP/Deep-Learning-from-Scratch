import numpy as np


# Step function (binary threshold)
def step_function(x):
    return np.array(x > 0, dtype=np.int8)

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU activation
def relu(x):
    return np.maximum(0, x)

# Identity function (no change)
def identity_function(x):
    return x

# Softmax function with overflow prevention
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # Overflow countermeasure
    return np.exp(x) / np.sum(np.exp(x))

# Mean Squared Error loss (version 1)
def mean_squared_error_1(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    return 0.5 * np.sum((y - t) ** 2) / batch_size

# Mean Squared Error loss (version 2)
def mean_squared_error_2(y, t):
    if y.ndim == 1:
        return 0.5 * np.sum((y - t) ** 2)
    batch_size = y.shape[0]
    return 0.5 * np.sum((y - t) ** 2) / batch_size

# Cross-Entropy Error loss (version 1)
def cross_entropy_error_1(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

# Cross-Entropy Error loss (version 2)
def cross_entropy_error_2(y, t):
    delta = 1e-7
    if y.ndim == 1:
        return -np.sum(t * np.log(y + delta))
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size

# Cross-Entropy Error for non-one-hot labels
def cross_entropy_error_3(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# Wrapper for MSE
def mean_squared_error(y, t):
    return mean_squared_error_2(y, t)

# Wrapper for CEE
def cross_entropy_error(y, t):
    return cross_entropy_error_2(y, t)