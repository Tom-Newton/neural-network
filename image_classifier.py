import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from network import Network, log_likelihood, transform_image, calculate_confusion_matrix
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
n_train = 1300
X_train = X[0: n_train, :]
X_test = X[n_train:, :]
Y_train = Y[0: n_train, :]
Y_test = Y[n_train:, :]




network = Network([[Softmax(number_classes, [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9)])],

                   [SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3),
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
                    SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3),
                                  (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)]),
                    SingleNeuron([(2, 0), (2, 1), (2, 2), (2, 3),
                                  (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)])],


                   [SingleNeuron([(3, 0), (3, 1), (3, 2), (3, 3),
                                  (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9)]),
                    SingleNeuron([(3, 0), (3, 1), (3, 2), (3, 3),
                                  (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9)]),
                    SingleNeuron([(3, 0), (3, 1), (3, 2), (3, 3),
                                  (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9)]),
                    SingleNeuron([(3, 0), (3, 1), (3, 2), (3, 3),
                                  (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9)]),
                    SingleNeuron([(3, 0), (3, 1), (3, 2), (3, 3),
                                  (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9)]),
                    SingleNeuron([(3, 0), (3, 1), (3, 2), (3, 3),
                                  (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9)]),
                    SingleNeuron([(3, 0), (3, 1), (3, 2), (3, 3),
                                  (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9)]),
                    SingleNeuron([(3, 0), (3, 1), (3, 2), (3, 3),
                                  (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9)]),
                    SingleNeuron([(3, 0), (3, 1), (3, 2), (3, 3),
                                  (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9)]),
                    SingleNeuron([(3, 0), (3, 1), (3, 2), (3, 3),
                                  (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9)])],


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

network.train(X_train, Y_train, 1)

predictions, x = network.update_network(X_train)
ll_train = log_likelihood(Y_train, predictions)
predictions, x = network.update_network(X_test)
ll_test = log_likelihood(Y_test, predictions)
print(
    f'Final ll_train = {ll_train/n_train}, ll_test = {ll_test/(n_total - n_train)}')

confusion_matrix = np.round(calculate_confusion_matrix(Y_test, predictions), 2)

for row in confusion_matrix:
    print(np.sum(row))

_, ax = plt.subplots()
ax.set_xticks(np.arange(number_classes))
ax.set_yticks(np.arange(number_classes))
ax.set_xticklabels(list(range(number_classes)))
ax.set_yticklabels(list(range(number_classes)))
ax.set_title('Confusion matrix P(predicted class | true class')
plt.xlabel('Predicted class')
plt.ylabel('True class')
for i in range(number_classes):
    for j in range(number_classes):
        ax.text(j, i, round(
            confusion_matrix[i, j], 3), ha='center', va='center', color='w')
ax.imshow(confusion_matrix)
plt.show()
