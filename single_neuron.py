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
        inputs.insert(0, np.ones(inputs[0].shape))
        self.X_tilde = np.concatenate(inputs, 1)

    def update_output(self):
        prediction = predict(self.X_tilde, self.w)
        self.output = prediction[0]
        self.x = prediction[1]


class Softmax(SingleNeuron):
    def __init__(self, input_locations):
        # Matrix with columns being w vectors
        self.W = None
        super().__init__(input_locations)

    def update_output(self):
        # Will be a vector of probabilities of each class
        prediction = softmax_predict(self.X_tilde, self.W)
        self.output = prediction


def logistic(x): return 1.0 / (1.0 + np.exp(-x))


def predict(X_tilde, w):
    x = np.dot(X_tilde, w)
    return logistic(x), x


def softmax_predict(X_tilde, W):
    # Returns a matrix: row for each x_tilde and column for each w
    a = np.exp(np.dot(X_tilde, (W - np.amax(W))))
    return a/(np.sum(a, axis=1)[:, np.newaxis])
