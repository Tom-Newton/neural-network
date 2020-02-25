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
        return self.data[0][0].output

    def output(self):
        # Assumes the network has been updated
        return self.data[0][0].output
