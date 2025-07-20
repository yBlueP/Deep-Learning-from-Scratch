import numpy as np

import common.layers as L
import common.functions as F


class ThreeLayerNet:
    def __init__(self):
        # structure
        self.layer1 = L.LinearLayer(2, 3)
        self.layer2 = L.LinearLayer(3, 2)
        self.layer3 = L.LinearLayer(2, 2, activation = F.identity_function)

        # init weight and bias
        self.layer1.init_weight(np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]))
        self.layer1.init_bias(np.array([0.1, 0.2, 0.3]))
        self.layer2.init_weight(np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]))
        self.layer2.init_bias(np.array([0.1, 0.2]))
        self.layer3.init_weight(np.array([[0.1, 0.3], [0.2, 0.4]]))
        self.layer3.init_bias(np.array([0.1, 0.2]))

    def predict(self, x):
        h1 = self.layer1.forward(x)
        h2 = self.layer2.forward(h1)
        h3 = self.layer3.forward(h2)

        return h3


if __name__ == "__main__":
    net = ThreeLayerNet()
    x = np.array([[1.0, 0.5]])
    y = net.predict(x)
    print(y)