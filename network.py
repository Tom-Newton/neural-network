import numpy as np
import single_neuron


def get_derivatives(network):
    derivatives = []
    for layer in network:
        derivatives.append([lambda y, X_tilde: 0]*len(layer))

    # def stem(y, X):
    #     return y - network[0][0].output()

    # stem is a function of y
    # derivatives are functions of y and X_tilde
    # y is relervant from the first stem. X_tilde only becomes relervant at the end

    def differentiate(network, stem=lambda y: (y - network[0][0].output()), i=0, j=0):
        neuron = network[i][j]
        derivatives[i][j] = lambda y, X_tilde: derivatives[i][j](
            y, X_tilde) + np.matmul(X_tilde.T, stem(y))

        for input_ in neuron.inputs:
            if type(input_) == tuple:
                def new_stem(y):
                    output = network[input_[0]][input_[1]].input_neuron.output()
                    return stem(y)*neuron.w[input_[1] + 1]*output*(np.ones(y.shape) - output)
                differentiate(network, new_stem, input_[0], input_[1])
    differentiate(network)
    return derivatives
