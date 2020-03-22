import numpy as np
from sklearn import datasets
from network import Network, log_likelihood, transform_image
from single_neuron import SingleNeuron, Softmax

digits = datasets.load_digits()
images = digits.images
labels = digits.target
number_classes = 10

X = np.zeros((images.shape[0], images.shape[1]*images.shape[2]))
for n, image in enumerate(images):
    X[n, :] = transform_image(images[n, :, :])


# Create Y matrix using onehot encoding
Y = np.zeros((labels.shape[0], number_classes))
for k in range(number_classes):
    Y[labels == k, k] = 1

n_total = X.shape[0]
n_train = 800
X_train = X[0: n_train, :]
X_test = X[n_train:, :]
Y_train = Y[0: n_train, :]
Y_test = Y[n_train:, :]


network = Network([[Softmax(number_classes, [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9)])],
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
                   [SingleNeuron(list(range(X.shape[1]))),
                    SingleNeuron(list(range(X.shape[1]))),
                    SingleNeuron(list(range(X.shape[1]))),
                    SingleNeuron(list(range(X.shape[1]))),
                    SingleNeuron(list(range(X.shape[1]))),
                    SingleNeuron(list(range(X.shape[1]))),
                    SingleNeuron(list(range(X.shape[1]))),
                    SingleNeuron(list(range(X.shape[1]))),
                    SingleNeuron(list(range(X.shape[1]))),
                    SingleNeuron(list(range(X.shape[1])))]])


predictions, x = network.update_network(X_train)
ll_train = log_likelihood(Y_train, predictions)
predictions, x = network.update_network(X_test)
ll_test = log_likelihood(Y_test, predictions)
print(
    f'Initial ll_train = {ll_train/n_train}, ll_test = {ll_test/(n_total - n_train)}')

network.train(X_train, Y_train, 10)

predictions, x = network.update_network(X_train)
ll_train = log_likelihood(Y_train, predictions)
predictions, x = network.update_network(X_test)
ll_test = log_likelihood(Y_test, predictions)
print(
    f'Final ll_train = {ll_train/n_train}, ll_test = {ll_test/(n_total - n_train)}')

