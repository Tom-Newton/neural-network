import numpy as np
import scipy.optimize
import single_neuron


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
            def new_derivative(Y, derivative=derivatives[i][j], stem=stem):
                return derivative(Y) + np.dot(self.data[i][j].X_tilde.T, stem(Y))
            derivatives[i][j] = new_derivative

            for input_location in self.data[i][j].input_locations:
                if type(input_location) == tuple:
                    def new_stem(Y, input_location=input_location):
                        output = self.data[input_location[0]
                                           ][input_location[1]].output
                        # TODO: Is there a nicer way to do this Manually fill [0, 0] then run function after that
                        if i == 0:
                            W = self.data[i][j].W
                            return (np.dot(Y, W.T)[:, input_location[1] + 1] - np.sum(self.data[0][0].output*W[input_location[1] + 1, :], axis=1))*output*(1 - output)
                        else:
                            return stem(Y)*self.data[i][j].w[input_location[1] + 1]*output*(1 - output)
                    differentiate(
                        new_stem, input_location[0], input_location[1])
        differentiate()
        return derivatives

    def update_network(self, X):
        for layer in self.data[::-1]:
            # Iterates layers in reverse order
            for neuron in layer:
                neuron.update_X_tilde(self.data, X)
                neuron.update_output()
        return self.data[0][0].output, self.data[0][0].x

    def output(self):
        # Assumes the network has been updated
        return self.data[0][0].output, self.data[0][0].x

    def train(self, X_train, Y_train, sigma0_squared, max_iterations=15000):
        derivatives = self.get_derivatives()
        # beta is all the w vectors stacked on top of each other

        def f(beta):
            # Update w s for every neuron
            self.unpack_beta(beta)

            predictions, x = self.update_network(X_train)
            f = -1*(log_likelihood(Y_train, predictions) -
                    log_prior_beta(beta, sigma0_squared))

            packed_derivatives = np.array([])
            for i, layer_derivatives in enumerate(derivatives):
                for neuron_derivative in layer_derivatives:
                    if i == 0:
                        for derivative in neuron_derivative(Y_train).T:
                            packed_derivatives = np.concatenate((packed_derivatives, derivative), axis=0)
                    else:
                        packed_derivatives = np.concatenate(
                            (packed_derivatives, neuron_derivative(Y_train)), axis=0)
            packed_derivatives = -1 * \
                (packed_derivatives - (beta/sigma0_squared))
            return f, packed_derivatives

        optimal = scipy.optimize.fmin_l_bfgs_b(
            f, self.pack_beta(), maxiter=max_iterations)
        if optimal[2]['warnflag'] == 0:
            return optimal[0]
        else:
            # raise ValueError(f'Search didn\'t converge. {optimal}')
            print(f'Search didn\'t converge. {optimal}')

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
        beta = np.array([])
        for i, layer in enumerate(self.data):
            for neuron in layer:
                if i == 0:
                    for w in neuron.W.T:
                        beta = np.concatenate((beta, w), axis=0)
                else:
                    beta = np.concatenate((beta, neuron.w), axis=0)
        return beta

class Convolutional(Network):
    def __init__(self, network_data, input_shape, network_input_shape):
        self.input_shape = input_shape
        self.network_input_shape = network_input_shape
        super().__init__(network_data)


def log_likelihood(Y, predictions):
    return np.sum(np.log(predictions)[Y == 1])


def log_prior_beta(beta, sigma0_squared):
    return -beta.shape[0]/2 * np.log(2*np.pi*sigma0_squared) - 0.5*np.dot(beta.T, beta)/sigma0_squared
