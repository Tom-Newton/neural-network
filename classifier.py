import numpy as np
import matplotlib.pyplot as plt
from sys import stdout

from network import Network, log_likelihood
from single_neuron import SingleNeuron

X = np.loadtxt('X.txt')
y = np.loadtxt('y.txt')

# Randomly permute the data
permutation = np.random.permutation(X.shape[0])
X = X[permutation, :]
y = y[permutation]

# Split the data into train and test sets. It has already be randomised by permuting the rows
n_train = 800
X_train = X[0: n_train, :]
X_test = X[n_train:, :]
y_train = y[0: n_train]
y_test = y[n_train:]

network = Network([[SingleNeuron([(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9)])],
                   [SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)]),
                    SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)]),
                    SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)]),
                    SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)]),
                    SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)]),
                    SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)]),
                    SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)]),
                    SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)]),
                    SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)]),
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


def plot_data_internal(X, y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    plt.figure()
    plt.xlim(xx.min(None), xx.max(None))
    plt.ylim(yy.min(None), yy.max(None))
    ax = plt.gca()
    ax.plot(X[y == 0, 0], X[y == 0, 1], 'ro', label='0')
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo', label='1')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Estimated P(yn = 1 | x)')
    plt.legend(loc='upper left', scatterpoints=1, numpoints=1)
    return xx, yy


def plot_predictive_distribution(X, y):
    xx, yy = plot_data_internal(X, y)
    # xx[n][m] yy[n][m] are coordinates for a square grid of 10000 points on the original data plot
    ax = plt.gca()
    X = np.concatenate((xx.ravel().reshape((-1, 1)),
                        yy.ravel().reshape((-1, 1))), 1)
    # Create X from the grid of points formed by concatenating xx and yy to form a list of 10000 [xx[n][m], yy[n][m]]
    Z = network.update_network(X)[0]
    Z = Z.reshape(xx.shape)
    cs2 = ax.contour(xx, yy, Z, cmap='RdBu', linewidths=2)
    plt.clabel(cs2, fmt='%2.1f', colors='k', fontsize=14)

predictions, x = network.update_network(X_train)
ll_train = log_likelihood(predictions, x, y_train)
predictions, x = network.update_network(X_test)
ll_test = log_likelihood(predictions, x, y_test)
print(f'Initial ll_train = {ll_train/n_train}, ll_test = {ll_test/(1000 - n_train)}')

network.train(X_train, y_train, 50)

predictions, x = network.update_network(X_train)
ll_train = log_likelihood(predictions, x, y_train)
predictions, x = network.update_network(X_test)
ll_test = log_likelihood(predictions, x, y_test)
print(f'Final ll_train = {ll_train/n_train}, ll_test = {ll_test/(1000 - n_train)}')

plot_predictive_distribution(X, y)
plt.show()
