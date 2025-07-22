import numpy as np

# x: one-dimensional array (batch size = 1)
def numerical_gradient_no_batch(f, x):
    gradient = np.zeros_like(x)
    h = 1e-4

    for idx in range(x.size):
        bak = x[idx]

        x[idx] = bak + h
        fxh1 = f(x)

        x[idx] = bak - h
        fxh2 = f(x)

        gradient[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = bak
    
    return gradient

# batch version
def numerical_gradient(f, x):
    if x.ndim == 1:
        return numerical_gradient_no_batch(f, x)
    
    grad = np.zeros_like(x)
    for idx, x in enumerate(x):
        grad[idx] = numerical_gradient_no_batch(f, x)
    return grad