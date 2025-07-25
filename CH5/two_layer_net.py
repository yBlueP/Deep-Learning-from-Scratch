import common.layers as L
import common.functions as F
from dataset.mnist import load_mnist

import numpy as np
import matplotlib.pyplot as plt


# TwoLayerNet class: Defines a simple two-layer neural network for MNIST classification
class TwoLayerNet:
    # Initialize the network with input, hidden, and output sizes, and weight initialization standard deviation
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # First linear layer with ReLU activation
        self.layer1 = L.LinearLayer(input_size, hidden_size, activation = L.ReLU(), weight_init_std = weight_init_std)
        # Second linear layer with ReLU activation (note: for output, identity might be better)
        self.layer2 = L.LinearLayer(hidden_size, output_size, activation = L.ReLU(), weight_init_std = weight_init_std)
        # Softmax with cross-entropy loss layer
        self.loss_layer = L.SoftmaxWithLoss()

    # Forward propagation: Pass input through layers
    # direct output result
    def forward(self, x):
        return self.layer2.forward(self.layer1.forward(x))
    
    # Compute loss: Forward pass + cross-entropy loss
    def loss(self, x, t):
        return self.loss_layer.forward(self.forward(x), t)
    
    # Compute accuracy: Compare predicted labels with true labels
    def accuracy(self, x, t):
        y = self.forward(x)

        # one-hot encoding to scalar
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        return np.sum(y == t) / x.shape[0]
    
    # Compute gradients using backpropagation
    def gradient(self, x, t):
        self.loss(x, t)

        # backward propagation
        dout = self.loss_layer.backward()
        
        # compare SoftmaxWithLoss.backward()
        with open("CH5/params.txt", "a") as f:
            f.write(f"dout: {dout}\n")

        dout = self.layer2.backward(dout)
        dout = self.layer1.backward(dout)

        return self.layer1.gradient_w, self.layer1.gradient_b, self.layer2.gradient_w, self.layer2.gradient_b
    

if __name__ == "__main__":
    # Load MNIST dataset with normalization and one-hot labels
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    # Lists to store training metrics
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # Training hyperparameters
    iters_num = 100
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.01
    iter_per_epoch = int(max(1, train_size / batch_size))

    # Initialize the network
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, weight_init_std=1)

    # Training loop
    for i in range(iters_num):
        for j in range(iter_per_epoch):
            # Select random batch
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            # Compute gradients
            network.gradient(x_batch, t_batch)

            # Update weights and biases using gradient descent
            network.layer1.W -= learning_rate * network.layer1.gradient_w
            network.layer1.b -= learning_rate * network.layer1.gradient_b
            network.layer2.W -= learning_rate * network.layer2.gradient_w
            network.layer2.b -= learning_rate * network.layer2.gradient_b

            # Compute and store loss for the batch
            loss = network.loss(x_batch, t_batch)
            train_loss_list.append(loss)
        
        # Compute and store accuracy at the end of each epoch
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        # Print progress
        print(f"epoch: {i}, train_acc: {train_acc}, test_acc: {test_acc}")

    # Plot loss curve
    plt.plot(range(len(train_loss_list)), train_loss_list)
    plt.xlabel("iterations (per 100)")
    plt.ylabel("loss")
    plt.show()

    # Plot accuracy curve
    plt.plot(range(len(test_acc_list)), test_acc_list, label="test acc")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend(loc="lower right")
    plt.show()
    
    
# epoch: 0, train_acc: 0.1023, test_acc: 0.0998
# epoch: 1, train_acc: 0.10151666666666667, test_acc: 0.0994
# epoch: 2, train_acc: 0.1012, test_acc: 0.0992
# epoch: 3, train_acc: 0.10138333333333334, test_acc: 0.0994
# epoch: 4, train_acc: 0.10133333333333333, test_acc: 0.0995
# epoch: 5, train_acc: 0.10175, test_acc: 0.0999
# epoch: 6, train_acc: 0.1032, test_acc: 0.1015
# epoch: 7, train_acc: 0.10611666666666666, test_acc: 0.1055
# epoch: 8, train_acc: 0.11133333333333334, test_acc: 0.1112
# epoch: 9, train_acc: 0.12811666666666666, test_acc: 0.1281
# epoch: 10, train_acc: 0.13508333333333333, test_acc: 0.137
# epoch: 11, train_acc: 0.1391, test_acc: 0.1398
# epoch: 12, train_acc: 0.14178333333333334, test_acc: 0.1428
# epoch: 13, train_acc: 0.14481666666666668, test_acc: 0.1458
# epoch: 14, train_acc: 0.14645, test_acc: 0.1478
# epoch: 15, train_acc: 0.1479, test_acc: 0.149
# epoch: 16, train_acc: 0.15036666666666668, test_acc: 0.1515
# epoch: 17, train_acc: 0.15235, test_acc: 0.1536
# epoch: 18, train_acc: 0.15425, test_acc: 0.1556
# epoch: 19, train_acc: 0.15805, test_acc: 0.159
# epoch: 20, train_acc: 0.1595, test_acc: 0.1607
# epoch: 21, train_acc: 0.15943333333333334, test_acc: 0.1605
# epoch: 22, train_acc: 0.16253333333333334, test_acc: 0.1639
# epoch: 23, train_acc: 0.1661, test_acc: 0.1679
# epoch: 24, train_acc: 0.16741666666666666, test_acc: 0.1689
# epoch: 25, train_acc: 0.17141666666666666, test_acc: 0.1721
# epoch: 26, train_acc: 0.17671666666666666, test_acc: 0.1768
# epoch: 27, train_acc: 0.1795, test_acc: 0.1793
# epoch: 28, train_acc: 0.18028333333333332, test_acc: 0.1805
# epoch: 29, train_acc: 0.1852, test_acc: 0.1851
# epoch: 30, train_acc: 0.1893, test_acc: 0.1886
# epoch: 31, train_acc: 0.19586666666666666, test_acc: 0.1939
# epoch: 32, train_acc: 0.19475, test_acc: 0.1935
# epoch: 33, train_acc: 0.20055, test_acc: 0.1993
# epoch: 34, train_acc: 0.20423333333333332, test_acc: 0.2021
# epoch: 35, train_acc: 0.20998333333333333, test_acc: 0.2097
# epoch: 36, train_acc: 0.21193333333333333, test_acc: 0.2117
# epoch: 37, train_acc: 0.21401666666666666, test_acc: 0.2139
# epoch: 38, train_acc: 0.21718333333333334, test_acc: 0.2175
# epoch: 39, train_acc: 0.22231666666666666, test_acc: 0.2225
# epoch: 40, train_acc: 0.22201666666666667, test_acc: 0.2225
# epoch: 41, train_acc: 0.22601666666666667, test_acc: 0.2274
# epoch: 42, train_acc: 0.23058333333333333, test_acc: 0.232
# epoch: 43, train_acc: 0.23463333333333333, test_acc: 0.2357
# epoch: 44, train_acc: 0.23485, test_acc: 0.2359
# epoch: 45, train_acc: 0.23795, test_acc: 0.2386
# epoch: 46, train_acc: 0.2396, test_acc: 0.2395
# epoch: 47, train_acc: 0.24283333333333335, test_acc: 0.2423
# epoch: 48, train_acc: 0.24335, test_acc: 0.2431
# epoch: 49, train_acc: 0.24373333333333333, test_acc: 0.2436
# epoch: 50, train_acc: 0.24545, test_acc: 0.2453
# epoch: 51, train_acc: 0.24958333333333332, test_acc: 0.2487
# epoch: 52, train_acc: 0.25155, test_acc: 0.2505
# epoch: 53, train_acc: 0.2529666666666667, test_acc: 0.2522
# epoch: 54, train_acc: 0.25266666666666665, test_acc: 0.2523
# epoch: 55, train_acc: 0.25225, test_acc: 0.2517
# epoch: 56, train_acc: 0.2564666666666667, test_acc: 0.255
# epoch: 57, train_acc: 0.26225, test_acc: 0.2601
# epoch: 58, train_acc: 0.2612, test_acc: 0.2586
# epoch: 59, train_acc: 0.2625, test_acc: 0.2599
# epoch: 60, train_acc: 0.26561666666666667, test_acc: 0.2633
# epoch: 61, train_acc: 0.2668833333333333, test_acc: 0.2645
# epoch: 62, train_acc: 0.27045, test_acc: 0.2682
# epoch: 63, train_acc: 0.2673833333333333, test_acc: 0.2646
# epoch: 64, train_acc: 0.26985, test_acc: 0.267
# epoch: 65, train_acc: 0.2706, test_acc: 0.2681
# epoch: 66, train_acc: 0.2703833333333333, test_acc: 0.2678
# epoch: 67, train_acc: 0.27355, test_acc: 0.2705
# epoch: 68, train_acc: 0.2750166666666667, test_acc: 0.2715
# epoch: 69, train_acc: 0.2742833333333333, test_acc: 0.2711
# epoch: 70, train_acc: 0.2727, test_acc: 0.2692
# epoch: 71, train_acc: 0.2744, test_acc: 0.2707
# epoch: 72, train_acc: 0.27465, test_acc: 0.2705
# epoch: 73, train_acc: 0.2758833333333333, test_acc: 0.2709
# epoch: 74, train_acc: 0.2759, test_acc: 0.2722
# epoch: 75, train_acc: 0.27736666666666665, test_acc: 0.2737
# epoch: 76, train_acc: 0.2768333333333333, test_acc: 0.2733
# epoch: 77, train_acc: 0.27771666666666667, test_acc: 0.2738
# epoch: 78, train_acc: 0.27831666666666666, test_acc: 0.2743
# epoch: 79, train_acc: 0.2784, test_acc: 0.2744
# epoch: 80, train_acc: 0.2795666666666667, test_acc: 0.2755
# epoch: 81, train_acc: 0.2786, test_acc: 0.2745
# epoch: 82, train_acc: 0.28041666666666665, test_acc: 0.2773
# epoch: 83, train_acc: 0.28028333333333333, test_acc: 0.2764
# epoch: 84, train_acc: 0.28013333333333335, test_acc: 0.2766
# epoch: 85, train_acc: 0.2796, test_acc: 0.2757
# epoch: 86, train_acc: 0.2811166666666667, test_acc: 0.278
# epoch: 87, train_acc: 0.2814333333333333, test_acc: 0.2785
# epoch: 88, train_acc: 0.2808333333333333, test_acc: 0.2774
# epoch: 89, train_acc: 0.28126666666666666, test_acc: 0.2784
# epoch: 90, train_acc: 0.28141666666666665, test_acc: 0.2785
# epoch: 91, train_acc: 0.28141666666666665, test_acc: 0.2783
# epoch: 92, train_acc: 0.2809333333333333, test_acc: 0.2779
# epoch: 93, train_acc: 0.2819833333333333, test_acc: 0.279
# epoch: 94, train_acc: 0.2819, test_acc: 0.2789
# epoch: 95, train_acc: 0.28271666666666667, test_acc: 0.28
# epoch: 96, train_acc: 0.2824, test_acc: 0.2795
# epoch: 97, train_acc: 0.2836166666666667, test_acc: 0.2812
# epoch: 98, train_acc: 0.2839833333333333, test_acc: 0.2816
# epoch: 99, train_acc: 0.28378333333333333, test_acc: 0.2811