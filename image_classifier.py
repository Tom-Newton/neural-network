import numpy as np
import cupy as cp
from network import Network, log_likelihood, calculate_confusion_matrix
from single_neuron import SingleNeuron, Softmax
from helper_functions import unpickle, normalise, un_transform_image

data_train = unpickle('cifar-10/data_batch_1')
data_test = unpickle('cifar-10/test_batch')

images_train = data_train[b'data']
images_test = data_test[b'data']
labels_train = np.array(data_train[b'labels'])
labels_test = np.array(data_test[b'labels'])
number_classes = 10

n_train = labels_train.shape[0]
n_test = labels_test.shape[0]

# Do a PCA
X_train = cp.array(normalise(images_train))
X_test = cp.array(normalise(images_test))
U, s, V = cp.linalg.svd(X_train)
n_components = 1000
X2_train = cp.zeros((n_train, n_components))
X2_test = cp.zeros((n_test, n_components))
for i in range(n_components):
    X2_train[:, i] = cp.dot(X_train, V[i, :])
    X2_test[:, i] = cp.dot(X_test, V[i, :])

# Create Y matrix using onehot encoding
Y_train = np.zeros((n_train, number_classes))
Y_test = np.zeros((n_test, number_classes))
for k in range(number_classes):
    Y_train[labels_train == k, k] = 1
    Y_test[labels_test == k, k] = 1

# Convert to cupy arrays
Y_train = cp.array(Y_train)
Y_test = cp.array(Y_test)

network = Network([[Softmax(number_classes, [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9)])],

                   [SingleNeuron([(2, i) for i in range(10)]),
                    SingleNeuron([(2, i) for i in range(10)]),
                    SingleNeuron([(2, i) for i in range(10)]),
                    SingleNeuron([(2, i) for i in range(10)]),
                    SingleNeuron([(2, i) for i in range(10)]),
                    SingleNeuron([(2, i) for i in range(10)]),
                    SingleNeuron([(2, i) for i in range(10)]),
                    SingleNeuron([(2, i) for i in range(10)]),
                    SingleNeuron([(2, i) for i in range(10)]),
                    SingleNeuron([(2, i) for i in range(10)])],

                   [SingleNeuron([(3, i) for i in range(10)]),
                    SingleNeuron([(3, i) for i in range(10)]),
                    SingleNeuron([(3, i) for i in range(10)]),
                    SingleNeuron([(3, i) for i in range(10)]),
                    SingleNeuron([(3, i) for i in range(10)]),
                    SingleNeuron([(3, i) for i in range(10)]),
                    SingleNeuron([(3, i) for i in range(10)]),
                    SingleNeuron([(3, i) for i in range(10)]),
                    SingleNeuron([(3, i) for i in range(10)]),
                    SingleNeuron([(3, i) for i in range(10)])],

                   [SingleNeuron(list(range(n_components))),
                    SingleNeuron(list(range(n_components))),
                    SingleNeuron(list(range(n_components))),
                    SingleNeuron(list(range(n_components))),
                    SingleNeuron(list(range(n_components))),
                    SingleNeuron(list(range(n_components))),
                    SingleNeuron(list(range(n_components))),
                    SingleNeuron(list(range(n_components))),
                    SingleNeuron(list(range(n_components))),
                    SingleNeuron(list(range(n_components)))]])


predictions, x = network.update_network(X2_train)
ll_train = log_likelihood(Y_train, predictions)
predictions, x = network.update_network(X2_test)
ll_test = log_likelihood(Y_test, predictions)
print(
    f'Initial ll_train = {ll_train/n_train}, ll_test = {ll_test/(n_test)}')

# To load previously trained weights
# w_data = np.load('cifar-10/w_data.npy', allow_pickle=True)
# network.set_weights(w_data)

network.train(X2_train, Y_train, 50)

predictions, x = network.update_network(X2_train)
ll_train = log_likelihood(Y_train, predictions)
predictions, x = network.update_network(X2_test)
ll_test = log_likelihood(Y_test, predictions)
print(
    f'Final ll_train = {ll_train/n_train}, ll_test = {ll_test/(n_test)}\n')

try:
    with open('cifar-10/ll_test.txt', 'r') as file:
        old_ll_test = float(file.readline())
except FileNotFoundError:
    old_ll_test = -np.inf

if old_ll_test <= ll_test:
    print('Updating old weights')
    np.save('cifar-10/w_data.npy', network.get_weights())
    with open('cifar-10/ll_test.txt', 'w+') as file:
        file.write(str(ll_test))
