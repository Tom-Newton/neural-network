import numpy as np
import matplotlib.pyplot as plt
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
X_train = normalise(images_train)
X_test = normalise(images_test)
U, s, V = np.linalg.svd(X_train)
print(s)
n_components = 200
X2_train = np.zeros((n_train, n_components))
X2_test = np.zeros((n_test, n_components))
for i in range(n_components):
    X2_train[:, i] = np.dot(X_train, V[i, :])
    X2_test[:, i] = np.dot(X_test, V[i, :])

# Create Y matrix using onehot encoding
Y_train = np.zeros((n_train, number_classes))
Y_test = np.zeros((n_test, number_classes))
for k in range(number_classes):
    Y_train[labels_train == k, k] = 1
    Y_test[labels_test == k, k] = 1

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

# w_data = np.load('w_data.npy', allow_pickle=True)
# network.set_weights(w_data)

network.train(X2_train, Y_train, 50)

predictions, x = network.update_network(X2_train)
ll_train = log_likelihood(Y_train, predictions)
predictions, x = network.update_network(X2_test)
ll_test = log_likelihood(Y_test, predictions)
print(
    f'Final ll_train = {ll_train/n_train}, ll_test = {ll_test/(n_test)}\n')

# confusion_matrix = np.round(calculate_confusion_matrix(Y_test, predictions), 2)

# _, ax = plt.subplots()
# ax.set_xticks(np.arange(number_classes))
# ax.set_yticks(np.arange(number_classes))
# ax.set_xticklabels(list(range(number_classes)))
# ax.set_yticklabels(list(range(number_classes)))
# ax.set_title('Confusion matrix P(predicted class | true class)')
# plt.xlabel('Predicted class')
# plt.ylabel('True class')
# for i in range(number_classes):
#     for j in range(number_classes):
#         ax.text(j, i, round(
#             confusion_matrix[i, j], 3), ha='center', va='center', color='w')
# ax.imshow(confusion_matrix)

# hard_predictions = np.argmax(predictions, 1)
# true_classes = np.argmax(Y_test, 1)

# for n in range(number_classes):
#     fig = plt.figure()
#     fig.suptitle(f'Supposed to be {n}s but incorrectly categorised')
#     incorrect_predictions = np.logical_and(
#         true_classes == n, hard_predictions != n)
#     incorrect_class = hard_predictions[incorrect_predictions]
#     x_indices = incorrect_predictions.nonzero()[0]
#     number_to_display = np.sum(incorrect_predictions)
#     for l in range(number_to_display):
#         plt.subplot(1, number_to_display, l+1)
#         plt.title(incorrect_class[l])
#         plt.imshow(un_transform_colour_image(images_test[x_indices[l]], (32, 32)))

with open('ll_test.txt', 'r') as file:
    old_ll_test = float(file.readline())
if old_ll_test <= ll_test:
    print('Updating old weights')
    np.save('w_data.npy', network.get_weights())
    with open('ll_test.txt', 'w+') as file:
        file.write(str(ll_test))

plt.show()
