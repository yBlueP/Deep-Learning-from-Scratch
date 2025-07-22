import numpy as np


# activation functions
def step_function(x):
    return np.array(x > 0, dtype=np.int8)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def identity_function(x):
    return x

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x


# loss functions
# standard version
def mean_squared_error_1(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    return 0.5 * np.sum((y - t) ** 2) / batch_size

def mean_squared_error_2(y, t):
    if y.ndim == 1:
        return 0.5 * np.sum((y - t) ** 2)
    batch_size = y.shape[0]
    return 0.5 * np.sum((y - t) ** 2) / batch_size

# standard version
def cross_entropy_error_1(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def cross_entropy_error_2(y, t):
    delta = 1e-7
    if y.ndim == 1:
        return -np.sum(t * np.log(y + delta))
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size

# non one-hot version
def cross_entropy_error_3(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def mean_squared_error(y, t):
    return mean_squared_error_2(y, t)

def cross_entropy_error(y, t):
    return cross_entropy_error_2(y, t)