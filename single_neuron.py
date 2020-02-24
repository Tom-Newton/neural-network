import numpy as np

# inputs is a list containing tuples:
# If the input_type is a neuron_output (i, j)
# If the input_type is a data_input n


class SingleNeuron:
    def __init__(self, input_locations):
        self.input_locations = input_locations
        self.w = np.random.randn(len(input_locations) + 1)
        self.X_tilde = None
        self.output = None

    def get_inputs(self, network, X):
        inputs = []
        for input_location in self.input_locations:
            # 1D arrays need to be converted to column vectors so they can be concatenated 
            if type(input_location) == tuple:
                inputs.append(network[input_location[0]]
                              [input_location[1]].output[:, np.newaxis])
            else:
                inputs.append(X[:, input_location][:, np.newaxis])
        return inputs

    def update_X_tilde(self, network, X):
        inputs = self.get_inputs(network, X)
        inputs.insert(0, np.ones(inputs[0].shape))
        self.X_tilde = np.concatenate(inputs, 1)

    def update_output(self):
        self.output = predict(self.X_tilde, self.w)


def logistic(x): return 1.0 / (1.0 + np.exp(-x))


def predict(X_tilde, w): return logistic(np.dot(X_tilde, w))
