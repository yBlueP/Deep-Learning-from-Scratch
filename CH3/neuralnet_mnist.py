import numpy as np
import pickle
import os

from dataset.mnist import load_mnist
import common.layers as L
import common.functions as F


def init_network():
    with open("CH3/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, flatten = True, one_hot_label = False)
    return x_train, t_train, x_test, t_test

class MnistNeuralNet:
    def __init__(self):
        # structure
        self.layer1 = L.LinearLayer(784, 50)
        self.layer2 = L.LinearLayer(50, 100)
        self.layer3 = L.LinearLayer(100, 10, activation = F.identity_function)

        self.params = init_network()

        # init weight and bias
        self.layer1.init_weight(self.params["W1"])
        self.layer1.init_bias(self.params["b1"])
        self.layer2.init_weight(self.params["W2"])
        self.layer2.init_bias(self.params["b2"])
        self.layer3.init_weight(self.params["W3"])
        self.layer3.init_bias(self.params["b3"])

    def predict(self, x):
        h1 = self.layer1.forward(x)
        h2 = self.layer2.forward(h1)
        h3 = self.layer3.forward(h2)

        return h3


if __name__ == "__main__":
    # three step: data preparation, model definition, model evaluation
    
    # data preparation
    _, _, x_test, t_test = get_data()

    # model definition
    network = MnistNeuralNet()

    # model evaluation
    accuracy_cnt = 0
    batch_size = 100
    for i in range(0, len(x_test), batch_size):
        y = network.predict(x_test[i: i + batch_size])
        p = np.argmax(y, axis = 1)
        accuracy_cnt += np.sum(p == t_test[i: i + batch_size])
    print("Accuracy:" + str(float(accuracy_cnt) / len(x_test)))