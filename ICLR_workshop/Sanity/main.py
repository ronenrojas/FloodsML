import numpy as np
import matplotlib.pyplot as plt

NUM_SAMPLES = 1000
X_DIM = 100


# Non-Linear
def f(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    return x1**2 + x2**3
    #return np.tan(x1) + np.exp(x2)


def create_dataset(n, d):
    x = np.random.normal(size=(n, d))
    b = np.random.binomial(1, 0.5, n)
    f1 = f(x[:, [0, 1]])
    f2 = f(x[:, [2, 3]])
    y = b*f1 + (1-b)*f2
    return x, y, b


if __name__ == "__main__":
    x, y, b = create_dataset(NUM_SAMPLES, X_DIM)
    idx0 = b == 0
    idx1 = b == 1
    fig, (ax1, ax2) = plt.subplots(2, 2)
    ax1[0].plot(x[idx0, 0], y[idx0], 'x')
    ax1[0].set_xlabel("x0")
    ax1[1].plot(x[idx0, 1], y[idx0], 'x')
    ax1[1].set_xlabel("x1")
    ax2[0].plot(x[idx0, 2], y[idx0], 'x')
    ax2[0].set_xlabel("x2")
    ax2[1].plot(x[idx0, 3], y[idx0], 'x')
    ax2[1].set_xlabel("x3")
    plt.show()


