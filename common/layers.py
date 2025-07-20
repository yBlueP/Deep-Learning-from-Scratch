import numpy as np

import common.functions as F


class LinearLayer:
    def __init__(self, in_size, out_size, activation = F.sigmoid):
        self.W = np.random.randn(in_size, out_size)
        self.b = np.random.randn(out_size)
        self.activation = activation

    def forward(self, x):
        return self.activation(np.dot(x, self.W) + self.b)

    def init_weight(self, w):
        self.W = w

    def init_bias(self, b):
        self.b = b
