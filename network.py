import numpy as np
import single_neuron


class Network:
    def __init__(self, network_data):
        self.data = network_data

    def get_derivatives(self):
        derivatives = []
        for layer in self.data:
            derivatives.append([lambda y: 0]*len(layer))

        # stem is a function of y
        # derivatives are functions of y the X_tilde s and the w s. X_dilde s
        # and w s are stored in the neuron objects. This makes the derivatives
        # functions of only y.

        def differentiate(network_data, stem=lambda y: (y - self.data[0][0].output), i=0, j=0):
            neuron = network_data[i][j]
            old_derivative = derivatives[i][j]

            derivatives[i][j] = lambda y: old_derivative(y) + np.dot(self.data[i][j].X_tilde.T, stem(y))

            for input_location in neuron.input_locations:
                if type(input_location) == tuple:
                    def new_stem(y):
                        output = self.data[input_location[0]
                                           ][input_location[1]].output
                        return stem(y)*self.data[i][j].w[input_location[1] + 1]*output*(np.ones(y.shape) - output)

                    differentiate(network_data, new_stem,
                                  input_location[0], input_location[1])
        differentiate(self.data)
        return derivatives

    def update_network(self, X):
        for layer in self.data[::-1]:
            # Iterates layers in reverse order
            for neuron in layer:
                neuron.update_X_tilde(self.data, X)
                neuron.update_output()
        return self.data[0][0].output

    def output(self):
        # Assumes the network has been updated
        return self.data[0][0].output
