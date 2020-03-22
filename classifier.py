import numpy as np
import matplotlib.pyplot as plt

from network import Network, log_likelihood
from single_neuron import SingleNeuron, Softmax

X = np.load('X2.np')
y = np.load('y2.np')

# Randomly permute the data
permutation = np.random.permutation(X.shape[0])
X = X[permutation, :]
y = y[permutation]
n_total = y.shape[0]
number_classes = 3

# Create Y matrix using onehot encoding
Y = np.zeros((X.shape[0], number_classes))
for k in range(number_classes):
    Y[y == k, k] = 1

# Split the data into train and test sets. It has already be randomised by permuting the rows
n_train = 800
X_train = X[0: n_train, :]
X_test = X[n_train:, :]
Y_train = Y[0: n_train, :]
Y_test = Y[n_train:, :]

network = Network([[Softmax(3, [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9)])],
                   [SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)]),
                    SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3),
                                  (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)]),
                    SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3),
                                  (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)]),
                    SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3),
                                  (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)]),
                    SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3),
                                  (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)]),
                    SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3),
                                  (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)]),
                    SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3),
                                  (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)]),
                    SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3),
                                  (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)]),
                    SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3),
                                  (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)]),
                    SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)])],
                   [SingleNeuron([0, 1]),
                    SingleNeuron([0, 1]),
                    SingleNeuron([0, 1]),
                    SingleNeuron([0, 1]),
                    SingleNeuron([0, 1]),
                    SingleNeuron([0, 1]),
                    SingleNeuron([0, 1]),
                    SingleNeuron([0, 1]),
                    SingleNeuron([0, 1]),
                    SingleNeuron([0, 1])]])

# network = Network([[SingleNeuron([(1, 0), (1, 1), (1, 2), (1, 3)])],
#                    [SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3)]), SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3)]),
#                     SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3)]), SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3)])],
#                    [SingleNeuron([0, 1]), SingleNeuron([0, 1]), SingleNeuron([0, 1]), SingleNeuron([0, 1])]])


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


predictions, x = network.update_network(X_train)
ll_train = log_likelihood(Y_train, predictions)
predictions, x = network.update_network(X_test)
ll_test = log_likelihood(Y_test, predictions)
print(
    f'Initial ll_train = {ll_train/n_train}, ll_test = {ll_test/(n_total - n_train)}')

network.train(X_train, Y_train, 8)

predictions, x = network.update_network(X_train)
ll_train = log_likelihood(Y_train, predictions)
predictions, x = network.update_network(X_test)
ll_test = log_likelihood(Y_test, predictions)
print(
    f'Final ll_train = {ll_train/n_train}, ll_test = {ll_test/(n_total - n_train)}')


plot_predictive_distribution(X, Y, lambda X: network.update_network(
    X)[0][:, 0], 'Predictive distribution P(y_n = 0 | x) with ML predictor')

plot_predictive_distribution(X, Y, lambda X: network.update_network(
    X)[0][:, 1], 'Predictive distribution P(y_n = 1 | x) with ML predictor')

plot_predictive_distribution(X, Y, lambda X: network.update_network(
    X)[0][:, 2], 'Predictive distribution P(y_n = 2 | x) with ML predictor')
plt.show()
