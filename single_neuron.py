import numpy as np
import cupy as cp

# inputs is a list containing tuples:
# If the input_type is a neuron_output (i, j)
# If the input_type is a data_input n


class SingleNeuron:
    def __init__(self, input_locations):
        # TODO: Make attributes with get functions private 
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
                                  ][input_location[1]].get_output()[:, np.newaxis]
        else:
            input_ = X[:, input_location][:, np.newaxis]
        # TODO: This should probably be done at the output from the convolutional neuron
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

    def get_X_tilde(self):
        return self.X_tilde

    def update_output(self, network_data, X):
        self._update_X_tilde(network_data, X)
        prediction = predict(self.get_X_tilde(), self.w)
        self.output = prediction[0]
        self.x = prediction[1]

    def get_output(self):
        return self.output

    def reset_weights(self):
        self.w = cp.random.randn(len(self.input_locations) + 1)

    def set_weights(self, w):
        self.w = w

    def get_weights(self):
        return self.w

    def get_new_derivative(self, stem, a, b):
        def new_derivative(Y):
            X_tilde = self.get_X_tilde()
            return cp.dot(X_tilde.T, stem(Y))
        return new_derivative

    def get_new_stem(self, network_data, input_location, stem, l):
        def new_stem(Y):
            # TODO: After removing `[:, np.newaxis]` from _get_input can remove the `[:, 0]`
            input_ = self._get_input(
                input_location, network_data, None)[:, 0]
            return stem(Y)*self.w[l]*input_*(1 - input_)
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
        prediction = softmax_predict(self.get_X_tilde(), self.W)
        self.output = prediction

    def reset_weights(self):
        self.W = cp.random.randn(
            len(self.input_locations) + 1, self.number_classes)

    def set_weights(self, W):
        self.W = W

    def get_weights(self):
        return self.W

    def get_new_stem(self, network_data, input_location, stem, l):
        def new_stem(Y):
            # TODO: After removing `[:, np.newaxis]` from _get_input can remove the `[:, 0]`
            input_ = self._get_input(input_location, network_data, None)[:, 0]
            return (cp.dot(Y, self.W.T)[:, l] - cp.sum(network_data[0][0].get_output()*self.W[l, :], axis=1))*input_*(1 - input_)
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
        self.X_tilde_array = None
        self.output_array = None

    def get_W_matrix(self):
        bias = self.w[0]
        W = cp.full(self.filter_shape, cp.nan)
        for row_number, i in enumerate(range(1, len(self.w), self.filter_shape[1])):
            W[row_number, :] = self.w[i: i + self.filter_shape[1]]
        return W, bias

    def _get_input(self, input_location, network_data, X):
        if type(input_location) == tuple:
            input_ = network_data[input_location[0]
                                  ][input_location[1]].get_output()
        else:
            input_ = X[:, input_location, :, :]
        if self.input_shape is None:
            self._set_input_dimensions(input_, input_location)
        return input_

    def _set_input_dimensions(self, input_, input_location):
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
        self.output_array = cp.full(self.output_shape, cp.nan)
        self.X_tilde_array = cp.full((self.output_shape[0],
                                      len(self.w),
                                      self.output_shape[1],
                                      self.output_shape[2]), cp.nan)

    def _map_inputs_for_convolution(self, input_section):
        return cp.concatenate([cp.array([1])] + [input_row for input_row in input_section], axis=0)

    def _update_X_tilde(self, network_data, X):
        input_ = self._get_inputs(network_data, X)[0]
        for a in range(self.output_shape[1]):
            for b in range(self.output_shape[2]):
                input_sections = input_[
                    :, a: a + self.filter_shape[0], b: b + self.filter_shape[1]]
                self.X_tilde_array[:, :, a, b] = cp.array([self._map_inputs_for_convolution(
                    input_section) for input_section in input_sections])

    def get_X_tilde(self, a=slice(None, None), b=slice(None, None)):
        return self.X_tilde_array[:, :, a, b]

    def update_output(self, network_data, X):
        self._update_X_tilde(network_data, X)
        for a in range(self.output_shape[1]):
            for b in range(self.output_shape[2]):
                prediction = predict(self.get_X_tilde(a, b), self.w)
                self.output_array[:, a, b] = prediction[0]

    def get_output(self, a=slice(None, None), b=slice(None, None)):
        return self.output_array[:, a, b]

    def reset_weights(self):
        self.w = cp.random.randn(1 + self.filter_shape[0]*self.filter_shape[1])

    def get_new_derivative(self, stem, a, b):
        def new_derivative(Y):
            X_tilde = self.get_X_tilde(a, b)
            return cp.dot(X_tilde.T, stem(Y))
        return new_derivative

    def get_new_stem(self, network_data, input_location, stem, a, b, l):
        def new_stem(Y):
            input_ = self._get_input(
                input_location, network_data, None)[:, a, b]
            return stem(Y)*self.w[l]*input_*(1 - input_)
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
