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

        # stem is a function of y
        # derivatives are functions of y the X_tilde s and the w s. X_dilde s
        # and w s are stored in the neuron objects. This makes the derivatives
        # functions of only y.

        def differentiate(stem=lambda y: (y - self.data[0][0].output), i=0, j=0):
            def new_derivative(y, derivative=derivatives[i][j], stem=stem):
                return derivative(y) + np.dot(self.data[i][j].X_tilde.T, stem(y))
            derivatives[i][j] = new_derivative

            for input_location in self.data[i][j].input_locations:
                if type(input_location) == tuple:
                    def new_stem(y, input_location=input_location):
                        output = self.data[input_location[0]
                                           ][input_location[1]].output
                        return stem(y)*self.data[i][j].w[input_location[1] + 1]*output*(np.ones(y.shape) - output)

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

    def train(self, X_train, y_train, sigma0_squared, max_iterations=15000):
        derivatives = self.get_derivatives()
        # beta is all the w vectors stacked on top of each other

        def f(beta):
            # Update w s for every neuron
            self.unpack_beta(beta)

            predictions, x = self.update_network(X_train)
            f = -1*(log_likelihood(predictions, x, y_train) - log_prior_beta(beta, sigma0_squared))

            packed_derivatives = np.array([])
            for layer_derivatives in derivatives:
                for neuron_derivative in layer_derivatives:
                    packed_derivatives = np.concatenate(
                        (packed_derivatives, neuron_derivative(y_train)), axis=0)
            packed_derivatives = -1*(packed_derivatives - (beta/sigma0_squared))
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
        for layer in self.data:
            for neuron in layer:
                neuron.w = beta[index:index + neuron.w.shape[0]]
                index += neuron.w.shape[0]

    def pack_beta(self):
        beta = np.array([])
        for layer in self.data:
            for neuron in layer:
                beta = np.concatenate((beta, neuron.w), axis=0)
        return beta


def log_likelihood(predictions, x, y):
    approximate_threshold = 20
    log_likelihood = 0
    for n in range(len(predictions)):
        # For ver large or very small x approximate ln(sigma(x))
        if x[n] < -approximate_threshold:
            log_likelihood += y[n]*x[n] + \
                (1 - y[n])*np.log(1.0 - predictions[n])
        elif x[n] > approximate_threshold:
            log_likelihood += y[n]*np.log(predictions[n]) + (1 - y[n])*(-x[n])
        else:
            log_likelihood += y[n]*np.log(predictions[n]) + \
                (1 - y[n])*np.log(1.0 - predictions[n])
    return log_likelihood


def log_prior_beta(beta, sigma0_squared):
    return -beta.shape[0]/2 * np.log(2*np.pi*sigma0_squared) - 0.5*np.dot(beta.T, beta)/sigma0_squared
