import numpy as np
import single_neuron


def get_derivatives(network):
    derivatives = []
    for layer in network:
        derivatives.append([lambda y, X_tilde: 0]*len(layer))

    # stem is a function of y
    # derivatives are functions of y and X_tilde
    # y is relervant from the first stem. X_tilde only becomes relervant at the end

    def differentiate(network, stem=lambda y: (y - network[0][0].output()), i=0, j=0):
        neuron = network[i][j]
        derivatives[i][j] = lambda y, X_tilde: derivatives[i][j](
            y, X_tilde) + np.matmul(X_tilde.T, stem(y))

        for input_location in neuron.input_locations:
            if type(input_location) == tuple:
                def new_stem(y):
                    output = network[input_location[0]
                                     ][input_location[1]].output
                    return stem(y)*neuron.w[input_location[1] + 1]*output*(np.ones(y.shape) - output)

                differentiate(network, new_stem,
                              input_location[0], input_location[1])
    differentiate(network)
    return derivatives


def update_network(network, X):
    for layer in network[::-1]:
        # Iterates layers in reverse order
        for neuron in layer:
            neuron.update_X_tilde(network, X)
            neuron.update_output()
    return network[0][0].output

def output(network):
    # Assumes the network has been updated
    return network[0][0].output
