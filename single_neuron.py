import numpy as np
import cupy as cp

# inputs is a list containing tuples:
# If the input_type is a neuron_output (i, j)
# If the input_type is a data_input n


class SingleNeuron:
    def __init__(self, input_locations):
        self.input_locations = input_locations
        self.w = cp.random.randn(len(self.input_locations) + 1)
        self.X_tilde = None
        self.output = None
        self.x = None

    def get_inputs(self, network_data, X):
        inputs = []
        for input_location in self.input_locations:
            try:
                # 1D arrays need to be converted to column vectors so they can be concatenated
                if type(input_location) == tuple:
                    inputs.append(network_data[input_location[0]]
                                  [input_location[1]].output[:, np.newaxis])
                else:
                    inputs.append(X[:, input_location][:, np.newaxis])
            except IndexError:
                raise IndexError(f'Incorrectly defined input locations. Input location {input_location} can\'t be '
                                 f'found from neuron {find_neuron_location(self, network_data)}')
        return inputs

    def update_X_tilde(self, network_data, X):
        inputs = self.get_inputs(network_data, X)
        inputs.insert(0, cp.ones(inputs[0].shape))
        self.X_tilde = cp.concatenate(inputs, 1)

    def update_output(self):
        prediction = predict(self.X_tilde, self.w)
        self.output = prediction[0]
        self.x = prediction[1]

    def reset_weights(self):
        self.w = cp.random.randn(len(self.input_locations) + 1)

    def set_weights(self, w):
        self.w = w

    def get_weights(self):
        return self.w

    def get_new_stem(self, network_data, input_location, stem):
        def new_stem(Y, input_location=input_location):
            # TODO: Can this be shifted back a level so we can get output from self.output
            output = network_data[input_location[0]][input_location[1]].output
            return stem(Y)*self.w[input_location[1] + 1]*output*(1 - output)
        return new_stem


class Softmax(SingleNeuron):
    def __init__(self, number_classes, input_locations):
        # Matrix with columns being w vectors for each class
        self.number_classes = number_classes
        super().__init__(input_locations)
        self.W = cp.random.randn(
            len(self.input_locations) + 1, self.number_classes)

    def update_output(self):
        # Will be a vector of probabilities of each class
        prediction = softmax_predict(self.X_tilde, self.W)
        self.output = prediction

    def reset_weights(self):
        self.W = cp.random.randn(
            len(self.input_locations) + 1, self.number_classes)

    def set_weights(self, W):
        self.W = W

    def get_weights(self):
        return self.W

    def get_new_stem(self, network_data, input_location, stem):
        def new_stem(Y):
            # TODO: Can this be shifted back a level so we can get output from self.output
            output = network_data[input_location[0]][input_location[1]].output
            return (cp.dot(Y, self.W.T)[:, input_location[1] + 1] - cp.sum(network_data[0][0].output*self.W[input_location[1] + 1, :], axis=1))*output*(1 - output)
        return new_stem


# TODO: Fix the occasional numerical error
def logistic(x):
    return 1.0 / (1.0 + cp.exp(-x))


def predict(X_tilde, w):
    x = cp.dot(X_tilde, w)
    # TODO: Get rid of extra x variable. It's no longer needed for computing log likelihood
    return logistic(x), x


def softmax_predict(X_tilde, W):
    # Returns a matrix: row with x_tilde for each n and column with w for each class
    a = cp.exp(cp.dot(X_tilde, (W - cp.amax(W))))
    return a/(cp.sum(a, axis=1)[:, np.newaxis])


def find_neuron_location(neuron_to_find, network_data):
    for i, layer in enumerate(network_data):
        for j, neuron in enumerate(layer):
            if neuron is neuron_to_find:
                return (i, j)
    return None
