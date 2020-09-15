import numpy as np
import cupy as cp
import scipy.optimize
from single_neuron import Softmax


class Network:
    def __init__(self, network_data):
        # 2D list of SingleNeuron objects
        self.data = network_data

    def get_derivatives(self):
        derivatives = []
        for layer in self.data:
            derivatives.append([lambda y: 0]*len(layer))

        # stem is a function of Y
        # derivatives are functions of Y the X_tilde s and the w s. X_tilde s
        # and w s are stored in the neuron objects. This makes the derivatives
        # functions of only Y.

        def differentiate(stem=lambda Y: (Y - self.data[0][0].output), i=0, j=0):

            # TODO: Put the new_derivative and new_stem functions in the corresponding neuron to get the right type
            def new_derivative(Y, derivative=derivatives[i][j], stem=stem):
                return derivative(Y) + cp.dot(self.data[i][j].X_tilde.T, stem(Y))
            derivatives[i][j] = new_derivative

            for input_location in self.data[i][j].input_locations:
                if type(input_location) == tuple:
                    neuron = self.data[i][j]
                    differentiate(
                        neuron.get_new_stem(self.data, input_location, stem), input_location[0], input_location[1])
        differentiate()
        return derivatives

    def update_network(self, X):
        for layer in self.data[::-1]:
            # Iterates layers in reverse order
            for neuron in layer:
                neuron.update_output(self.data, X)
        return self.data[0][0].output, self.data[0][0].x

    def output(self):
        # Assumes the network has been updated
        return self.data[0][0].output, self.data[0][0].x

    def reset_weights(self):
        for layer in self.data:
            for neuron in layer:
                neuron.reset_weights()

    def set_weights(self, w_data):
        for w_layer, layer in zip(w_data, self.data):
            for w, neuron in zip(w_layer, layer):
                neuron.set_weights(w)

    def get_weights(self):
        w_data = []
        for layer in self.data:
            w_layer = []
            for neuron in layer:
                if type(neuron) == Softmax:
                    w_layer.append(neuron.W)
                else:
                    w_layer.append(neuron.w)
            w_data.append(w_layer)
        return w_data

    def train(self, X_train, Y_train, sigma0_squared, max_iterations=100000):
        derivatives = self.get_derivatives()
        # beta is all the w vectors stacked on top of each other

        def f(beta):
            beta = cp.array(beta)
            # Update w s for every neuron
            self.unpack_beta(beta)

            predictions, x = self.update_network(X_train)
            f = -1*(log_likelihood(Y_train, predictions) +
                    log_prior_beta(beta, sigma0_squared))
            return f

        def gradient_f(beta):
            beta = cp.array(beta)
            # Update w s for every neuron
            self.unpack_beta(beta)

            packed_derivatives = cp.array([])
            for i, layer_derivatives in enumerate(derivatives):
                for neuron_derivative in layer_derivatives:
                    if i == 0:
                        for derivative in neuron_derivative(Y_train).T:
                            packed_derivatives = cp.concatenate(
                                (packed_derivatives, derivative), axis=0)
                    else:
                        packed_derivatives = cp.concatenate(
                            (packed_derivatives, neuron_derivative(Y_train)), axis=0)
            packed_derivatives = -1 * \
                (packed_derivatives - (beta/sigma0_squared))
            return cp.asnumpy(packed_derivatives)

        optimal = scipy.optimize.fmin_l_bfgs_b(
            f, cp.asnumpy(self.pack_beta()), fprime=gradient_f, pgtol=1, maxfun=max_iterations, maxiter=max_iterations)

        if optimal[2]['warnflag'] != 0:
            print(
                f'Search didn\'t converge. warnflag {optimal[2]["warnflag"]}')
            if optimal[2]['warnflag'] == 2:
                print(optimal[2]['warnflag']['task'])

        return optimal[0]

    def unpack_beta(self, beta):
        index = 0
        for i, layer in enumerate(self.data):
            for neuron in layer:
                if i == 0:
                    for k, w in enumerate(neuron.W.T):
                        neuron.W[:, k] = beta[index:index + w.shape[0]]
                        index += w.shape[0]
                else:
                    neuron.w = beta[index:index + neuron.w.shape[0]]
                    index += neuron.w.shape[0]

    def pack_beta(self):
        beta = cp.array([])
        for i, layer in enumerate(self.data):
            for neuron in layer:
                if i == 0:
                    for w in neuron.W.T:
                        beta = cp.concatenate((beta, w), axis=0)
                else:
                    beta = cp.concatenate((beta, neuron.w), axis=0)
        return beta

class Convolutional(Network):
    def __init__(self, network_data, input_shape, network_input_shape):
        self.input_shape = input_shape
        self.network_input_shape = network_input_shape
        super().__init__(network_data)


def log_prior_beta(beta, sigma0_squared):
    return -beta.shape[0]/2 * cp.log(2*np.pi*sigma0_squared) - 0.5*cp.dot(beta.T, beta)/sigma0_squared


def log_likelihood(Y, predictions):
    return cp.sum(cp.log(predictions)[Y == 1])


def calculate_confusion_matrix(Y, predictions):
    hard_predictions = np.argmax(predictions, 1)
    true_classes = np.argmax(Y, 1)
    matrix = np.zeros((Y.shape[1], Y.shape[1]))
    for i in range(matrix.shape[0]):
        # i is true class j is predicted class
        for j in range(matrix.shape[1]):
            matrix[i, j] = np.sum(np.logical_and(
                true_classes == i, hard_predictions == j), axis=0)/np.sum(true_classes == i, axis=0)
    return matrix
