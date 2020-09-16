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

    def _get_input(self, input_location, network_data, X):
        # 1D arrays need to be converted to column vectors so they can be concatenated
        # TODO: Can the [:, np.newaxis] be avoided using cp.array(inputs).T? Is this faster?
        if type(input_location) == tuple:
            input_ = network_data[input_location[0]
                                  ][input_location[1]].output[:, np.newaxis]
        else:
            input_ = X[:, input_location][:, np.newaxis]
        if len(input_.shape) > 2:
            if (input_.shape[1], input_.shape[2]) == (1, 1):
                return input_[:, 0, 0]
            raise ValueError('SingleNeuron given 2D arrays as an input. scalars were expected. '
                             f'input_location = {input_location} input shape = {input_.shape}')
        return input_

    def _get_inputs(self, network_data, X):
        inputs = []
        for input_location in self.input_locations:
            try:
                inputs.append(self._get_input(input_location, network_data, X))
            except IndexError:
                raise IndexError(f'Incorrectly defined input locations. Input location {input_location} can\'t be '
                                 f'found from neuron {find_neuron_location(self, network_data)}')
        return inputs

    def _update_X_tilde(self, network_data, X):
        inputs = self._get_inputs(network_data, X)
        inputs.insert(0, cp.ones(inputs[0].shape))
        self.X_tilde = cp.concatenate(inputs, 1)

    def update_output(self, network_data, X):
        self._update_X_tilde(network_data, X)
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
            # TODO: After removing `[:, np.newaxis]` from _get_input can remove the `[:, 0]`
            input_ = self._get_input(input_location, network_data, None)[:, 0]
            return stem(Y)*self.w[input_location[1] + 1]*input_*(1 - input_)
        return new_stem


class Softmax(SingleNeuron):
    def __init__(self, number_classes, input_locations):
        self.number_classes = number_classes
        super().__init__(input_locations)
        # Matrix with columns being w vectors for each class
        self.W = cp.random.randn(
            len(self.input_locations) + 1, self.number_classes)

    def update_output(self, network_data, X):
        self._update_X_tilde(network_data, X)
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

    def get_new_stem(self, network_data, input_location, _):
        def new_stem(Y):
            # TODO: After removing `[:, np.newaxis]` from _get_input can remove the `[:, 0]`
            input_ = self._get_input(input_location, network_data, None)[:, 0]
            return (cp.dot(Y, self.W.T)[:, input_location[1] + 1] - cp.sum(network_data[0][0].output*self.W[input_location[1] + 1, :], axis=1))*input_*(1 - input_)
        return new_stem


class Convolutional(SingleNeuron):
    def __init__(self, input_location, filter_shape):
        # Only allow a single 2D input.
        self.input_location = input_location
        super().__init__(input_locations=[input_location])
        self.filter_shape = filter_shape
        self.input_shape = None
        self.output_shape = None
        # 1D stack of 2D weight matrix. Stored 1D so they can be easily used in SingleNeuron
        self.w = cp.random.randn(1 + self.filter_shape[0]*self.filter_shape[1])
        self.output = None

    def get_W_matrix(self):
        bias = self.w[0]
        W = cp.full(self.filter_shape, cp.nan)
        for row_number, i in enumerate(range(1, len(self.w), self.filter_shape[1])):
            W[row_number, :] = self.w[i: i + self.filter_shape[1]]
        return W, bias

    def _get_input(self, input_location, network_data, X):
        if type(input_location) == tuple:
            input_ = network_data[input_location[0]][input_location[1]].output
        else:
            input_ = X[:, input_location, :, :]
        self.input_shape = input_.shape
        if len(self.input_shape) < 3:
            raise ValueError('Convolutional given scalars an input. 2D arrays were expected. '
                             f'input_location = {input_location} input shape = {input_.shape}')
        self.output_shape = (
            self.input_shape[0],
            self.input_shape[1] - (self.filter_shape[0] - 1),
            self.input_shape[2] - (self.filter_shape[1] - 1))
        if 0 in self.output_shape:
            raise ValueError(f'Impossible filter_shape. The given filter_shape {self.filter_shape} with input_shape '
                             f'{self.input_shape} would give an output_shape {self.output_shape} which includes a 0, '
                             'and is therefore not possible')
        self.output = cp.full(self.output_shape, cp.nan)
        return input_

    def _map_inputs_for_convolution(self, input_section):
        return cp.concatenate([cp.array([1])] + [input_row for input_row in input_section], axis=0)

    def _update_X_tilde(self, input_, a, b):
        input_sections = input_[
            :, a: a + self.filter_shape[0], b: b + self.filter_shape[1]]
        self.X_tilde = cp.array([self._map_inputs_for_convolution(
            input_section) for input_section in input_sections])

    def update_output(self, network_data, X):
        input_ = self._get_inputs(network_data, X)[0]
        for a in range(self.output_shape[1]):
            for b in range(self.output_shape[2]):
                self._update_X_tilde(input_, a, b)
                prediction = predict(self.X_tilde, self.w)
                self.output[:, a, b] = prediction[0]

    def reset_weights(self):
        self.w = cp.random.randn(1 + self.filter_shape[0]*self.filter_shape[1])

    def get_new_stem(self, network_data, input_location, stem):
        print('used convolutional stem')

        def new_stem(Y, input_location=input_location):
            print('used convolutional new_stem')
            output = network_data[input_location[0]
                                  ][input_location[1]].output[:, 0, 0]
            return stem(Y)*self.w[input_location[1] + 1]*output*(1 - output)
        return new_stem

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


def find_neuron_location(neuron_to_find, network_data):
    for i, layer in enumerate(network_data):
        for j, neuron in enumerate(layer):
            if neuron is neuron_to_find:
                return (i, j)
    return None
