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


class Softmax(SingleNeuron):
    def __init__(self, number_classes, input_locations):
        self.number_classes = number_classes
        super().__init__(input_locations)
        # Matrix with columns being w vectors for each class
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


class Convolutional:
    def __init__(self, input_locations):
        self.input_locations = input_locations
        # Matrix of 2D weight matrix
        self.W = cp.random.randn(self.input_locations.shape)
        self.X_tilde = None
        self.output = None
        self.x = None

    def get_inputs(self, )

# TODO: Add a pooling/subsampling neuron


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
