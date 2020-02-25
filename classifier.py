import numpy as np
import matplotlib.pyplot as plt
from sys import stdout

from network import Network
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


def compute_average_ll(X, y):
    # Vector of P(yn = 1) for each datapoint
    output_prob = network.update_network(X)
    return np.mean(y * np.log(output_prob) + (1 - y) * np.log(1.0 - output_prob))


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
    Z = network.update_network(X)
    Z = Z.reshape(xx.shape)
    cs2 = ax.contour(xx, yy, Z, cmap='RdBu', linewidths=2)
    plt.clabel(cs2, fmt='%2.1f', colors='k', fontsize=14)
    plt.show()


def plot_ll(ll):
    plt.figure()
    ax = plt.gca()
    plt.xlim(0, len(ll) + 2)
    plt.ylim(min(ll) - 0.1, max(ll) + 0.1)
    ax.plot(np.arange(1, len(ll) + 1), ll, 'r-')
    plt.xlabel('Steps')
    plt.ylabel('Average log-likelihood')
    plt.title('Plot Average Log-likelihood Curve')
    plt.show()


def gradient_descent(X_train, y_train, X_test, y_test, number_steps, learning_rate):
    ll_train = np.zeros(number_steps)
    ll_test = np.zeros(number_steps)
    for n in range(number_steps):
        network.update_network(X_train)
        derivatives = network.get_derivatives()

        for i, layer in enumerate(network.data):
            for j, neuron in enumerate(layer):
                gradient = derivatives[i][j](y_train)
                neuron.w += learning_rate*gradient

        ll_train[n] = compute_average_ll(X_train, y_train)
        ll_test[n] = compute_average_ll(X_test, y_test)

        if n % 100 == 0:
            stdout.write(f'\t{int(100 * n/number_steps)}%\tll_train = {ll_train[n]}\t\tll_test = {ll_test[n]}\r')
            stdout.flush()

    return ll_train, ll_test


learning_rate = 0.0004
number_steps = 25000

ll_train, ll_test = gradient_descent(X_train, y_train, X_test,
                                     y_test, number_steps, learning_rate)

print(f'Initial ll_train = {ll_train[0]}, ll_test = {ll_test[0]}')
print(f'Final ll_train = {ll_train[-1]}, ll_test = {ll_test[-1]}')

plot_ll(ll_train)
plot_ll(ll_test)
plot_predictive_distribution(X, y)
