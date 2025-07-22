import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    x0 = np.arange(-3.0, 3.0, 0.1)
    x1 = np.arange(-3.0, 3.0, 0.1)
    X, Y = np.meshgrid(x0, x1)
    Z = X**2 + Y**2  # 计算函数值 f(x0, x1) = x0^2 + x1^2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_zlabel('f(x)')
    plt.show()

