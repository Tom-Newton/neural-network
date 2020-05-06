import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from network import Network, log_likelihood, calculate_confusion_matrix
from single_neuron import SingleNeuron, Softmax

digits = datasets.load_digits()
images = digits.images
labels = digits.target
number_classes = 10


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

def normalise(x):
    n_data = x.shape[0]
    m = x.sum(axis=0)
    x0 = x - m/n_data
    s = np.sqrt((x0**2).sum(axis=0)/n_data)
    ss = np.array([ tmp if tmp != 0 else 1 for tmp in s])
    x00 = x0 / ss
    return x00

X = np.zeros((images.shape[0], images.shape[1]*images.shape[2]))
for n, image in enumerate(images):
    X[n, :] = transform_image(images[n, :, :])

n_total = X.shape[0]
n_train = 1300

X = normalise(X)
U, s, V = np.linalg.svd(X)

n_components = 50
X2 = np.zeros((n_total, n_components))
for i in range(n_components):
    X2[:, i] = np.dot(X, V[i, :])

# Create Y matrix using onehot encoding
Y = np.zeros((labels.shape[0], number_classes))
for k in range(number_classes):
    Y[labels == k, k] = 1

X_train = X2[0: n_train, :]
X_test = X2[n_train:, :]
Y_train = Y[0: n_train, :]
Y_test = Y[n_train:, :]

network = Network([[Softmax(number_classes, [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9)])],

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


predictions, x = network.update_network(X_train)
ll_train = log_likelihood(Y_train, predictions)
predictions, x = network.update_network(X_test)
ll_test = log_likelihood(Y_test, predictions)
print(
    f'Initial ll_train = {ll_train/n_train}, ll_test = {ll_test/(n_total - n_train)}')

# w_data = np.load('w_data.npy', allow_pickle=True)
# network.set_weights(w_data)

network.train(X_train, Y_train, 0.4)

predictions, x = network.update_network(X_train)
ll_train = log_likelihood(Y_train, predictions)
predictions, x = network.update_network(X_test)
ll_test = log_likelihood(Y_test, predictions)
print(
    f'Final ll_train = {ll_train/n_train}, ll_test = {ll_test/(n_total - n_train)}\n')

confusion_matrix = np.round(calculate_confusion_matrix(Y_test, predictions), 2)

_, ax = plt.subplots()
ax.set_xticks(np.arange(number_classes))
ax.set_yticks(np.arange(number_classes))
ax.set_xticklabels(list(range(number_classes)))
ax.set_yticklabels(list(range(number_classes)))
ax.set_title('Confusion matrix P(predicted class | true class)')
plt.xlabel('Predicted class')
plt.ylabel('True class')
for i in range(number_classes):
    for j in range(number_classes):
        ax.text(j, i, round(
            confusion_matrix[i, j], 3), ha='center', va='center', color='w')
ax.imshow(confusion_matrix)

hard_predictions = np.argmax(predictions, 1)
true_classes = np.argmax(Y_test, 1)

for n in range(number_classes):
    fig = plt.figure()
    fig.suptitle(f'Supposed to be {n}s but incorrectly categorised')
    incorrect_predictions = np.logical_and(
        true_classes == n, hard_predictions != n)
    incorrect_class = hard_predictions[incorrect_predictions]
    x_indices = incorrect_predictions.nonzero()[0]
    number_to_display = np.sum(incorrect_predictions)
    for l in range(number_to_display):
        plt.subplot(1, number_to_display, l+1)
        plt.title(incorrect_class[l])
        plt.imshow(images[n_train + x_indices[l]])

with open('ll_test.txt', 'r') as file:
    old_ll_test = float(file.readline())
if old_ll_test <= ll_test:
    print('Updating old weights')
    np.save('w_data.npy', network.get_weights())
    with open('ll_test.txt', 'w') as file:
        file.write(str(ll_test))

plt.show()
