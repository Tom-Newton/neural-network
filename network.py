import numpy as np
import single_neuron

def get_derivatives(network):
    derivatives = []
    for layer in network:
        derivatives.append(['']*len(layer))
    def differentiate(network, stem='(y - sigma(X00w00))', i=0, j=0):
        neuron = network[i][j]
        derivatives[i][j] += f'+ (X{i}{j})T {stem}'

        for input_ in neuron.inputs:
            if type(input_) == tuple:
                new_stem = stem + f'(w{i}{j}){input_[1] + 1} sigma(X{input_[0]}{input_[1]})(1 - sigma(X{input_[0]}{input_[1]}))'
                differentiate(network, new_stem, input_[0], input_[1])
    differentiate(network)
    return derivatives