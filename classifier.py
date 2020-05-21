import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from network import Network, log_likelihood
from single_neuron import SingleNeuron, Softmax
from helper_functions import plot_predictive_distribution, plot_data_internal

X = np.load('basic/X2.np')
y = np.load('basic/y2.np')

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

# Convert to cupy arrays
X = cp.array(X)
Y = cp.array(Y)

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


predictions, x = network.update_network(X_train)
ll_train = log_likelihood(Y_train, predictions)
predictions, x = network.update_network(X_test)
ll_test = log_likelihood(Y_test, predictions)
print(
    f'Initial ll_train = {ll_train/n_train}, ll_test = {ll_test/(n_total - n_train)}')

network.train(X_train, Y_train, 2)

predictions, x = network.update_network(X_train)
ll_train = log_likelihood(Y_train, predictions)
predictions, x = network.update_network(X_test)
ll_test = log_likelihood(Y_test, predictions)
print(
    f'Final ll_train = {ll_train/n_train}, ll_test = {ll_test/(n_total - n_train)}')

# TODO: This is really nasty
plot_predictive_distribution(cp.asnumpy(X), cp.asnumpy(Y), lambda X: cp.asnumpy(network.update_network(
    cp.array(X))[0][:, 0]), 'Predictive distribution P(y_n = 0 | x) with ML predictor')

plot_predictive_distribution(cp.asnumpy(X), cp.asnumpy(Y), lambda X: cp.asnumpy(network.update_network(
    cp.array(X))[0][:, 1]), 'Predictive distribution P(y_n = 1 | x) with ML predictor')

plot_predictive_distribution(cp.asnumpy(X), cp.asnumpy(Y), lambda X: cp.asnumpy(network.update_network(
    cp.array(X))[0][:, 2]), 'Predictive distribution P(y_n = 2 | x) with ML predictor')
plt.show()
