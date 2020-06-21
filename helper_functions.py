import numpy as np
import pickle
import matplotlib.pyplot as plt


def transform_image(image):
    x = np.array([])
    for row in image:
        x = np.concatenate((x, row), axis=0)
    return x


def un_transform_image(x, shape):
    image = np.zeros(shape)
    index = 0
    for i in range(shape[0]):
        image[i, :] = x[index: index + shape[1]]
        index += shape[1]
    return image


def un_transform_colour_image(x, shape):
    n = shape[0]*shape[1]
    red = un_transform_image(x[:n], shape)
    green = un_transform_image(x[n:2*n], shape)
    blue = un_transform_image(x[2*n:], shape)
    return np.dstack((red, green, blue))


def normalise(x):
    n_data = x.shape[0]
    m = x.sum(axis=0)
    x0 = x - m/n_data
    s = np.sqrt((x0**2).sum(axis=0)/n_data)
    ss = np.array([tmp if tmp != 0 else 1 for tmp in s])
    x00 = x0 / ss
    return x00


def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

def plot_data_internal(X, Y, title):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    plt.figure()
    plt.xlim(xx.min(None), xx.max(None))
    plt.ylim(yy.min(None), yy.max(None))
    ax = plt.gca()
    ax.plot(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], 'ro', label='0')
    ax.plot(X[Y[:, 1] == 1, 0], X[Y[:, 1] == 1, 1], 'bo', label='1')
    ax.plot(X[Y[:, 2] == 1, 0], X[Y[:, 2] == 1, 1], 'go', label='2')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.legend(loc='upper left', scatterpoints=1, numpoints=1)
    return xx, yy


def plot_predictive_distribution(X, Y, predictor, title):
    xx, yy = plot_data_internal(X, Y, title)
    # xx[n][m] yy[n][m] are coordinates for a square grid of 10000 points on the original data plot
    ax = plt.gca()
    X = np.concatenate((xx.ravel().reshape((-1, 1)),
                        yy.ravel().reshape((-1, 1))), 1)
    # Create X from the grid of points formed by concatenating xx and yy to form a list of 10000 [xx[n][m], yy[n][m]]
    Z = predictor(X)
    Z = Z.reshape(xx.shape)
    cs2 = ax.contour(xx, yy, Z, 20, cmap='RdBu', linewidths=1)
    plt.clabel(cs2, fmt='%2.1f', colors='k', fontsize=8)