import numpy as np

import common.functions as F


class LinearLayer:
    def __init__(self, in_size, out_size, activation = F.sigmoid, weight_init_std = 0.01):
        self.W = np.random.randn(in_size, out_size) * weight_init_std
        self.b = np.random.randn(out_size) * weight_init_std
        self.activation = activation
        self.gradient_w = None
        self.gradient_b = None

    def forward(self, x):
        return self.activation(np.dot(x, self.W) + self.b)

    def init_weight(self, w):
        self.W = w

    def init_bias(self, b):
        self.b = b
