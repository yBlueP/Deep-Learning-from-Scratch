import common.layers as L
import common.functions as F
import common.gradient as G
from dataset.mnist import load_mnist

import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.layer1 = L.LinearLayer(input_size, hidden_size, weight_init_std = weight_init_std)
        self.layer2 = L.LinearLayer(hidden_size, output_size, activation = F.softmax, weight_init_std = weight_init_std)
    
    def forward(self, x):
        return self.layer2.forward(self.layer1.forward(x))
    
    def loss(self, x, t):
        y = self.forward(x)
        return F.cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.forward(x)

        # one-hot encoding to scalar
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        return np.sum(y == t) / x.shape[0]
    
    def numerical_gradient(self, x, t):
        loss_w = lambda W: self.loss(x, t)

        self.layer1.gradient_w = G.numerical_gradient(loss_w, self.layer1.W)
        self.layer1.gradient_b = G.numerical_gradient(loss_w, self.layer1.b)
        self.layer2.gradient_w = G.numerical_gradient(loss_w, self.layer2.W)
        self.layer2.gradient_b = G.numerical_gradient(loss_w, self.layer2.b)

        return self.layer1.gradient_w, self.layer1.gradient_b, self.layer2.gradient_w, self.layer2.gradient_b
    

if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iters_num = 5
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    iter_per_epoch = int(max(1, train_size / batch_size))

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    for i in range(iters_num):
        for j in range(iter_per_epoch):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            grad_w1, grad_b1, grad_w2, grad_b2 = network.numerical_gradient(x_batch, t_batch)

            network.layer1.W -= learning_rate * grad_w1
            network.layer1.b -= learning_rate * grad_b1
            network.layer2.W -= learning_rate * grad_w2
            network.layer2.b -= learning_rate * grad_b2

            loss = network.loss(x_batch, t_batch)
            train_loss_list.append(loss)
        
        # train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)

        # train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        # print(f"epoch: {i}, train_acc: {train_acc}, test_acc: {test_acc}")
        print(f"epoch: {i}, test_acc: {test_acc}")

    plt.plot(range(len(train_loss_list)), train_loss_list)
    plt.xlabel("iterations (per 100)")
    plt.ylabel("loss")
    plt.show()

    # plt.plot(range(len(train_acc_list)), train_acc_list, label="train acc")
    plt.plot(range(len(test_acc_list)), test_acc_list, label="test acc")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend(loc="lower right")
    plt.show()
    