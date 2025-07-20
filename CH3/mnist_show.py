from dataset.mnist import load_mnist

from matplotlib import pyplot as plt

if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False)
    
    img = x_train[0]
    label = t_train[0]

    print("label = ", label)
    print("img.shape = ", img.shape)

    img = img.reshape(28, 28)
    print("reshaped img = ", img.shape)

    plt.imshow(img, cmap = "gray")
    plt.show()