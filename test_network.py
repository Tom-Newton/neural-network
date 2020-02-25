import unittest
import numpy as np
from network import Network
from single_neuron import SingleNeuron, predict, logistic


class Tests(unittest.TestCase):
    def setUp(self):
        self.network = Network([[SingleNeuron([(1, 0), (1, 1)])],
                                [SingleNeuron([(2, 0), (2, 1)]),
                                 SingleNeuron([(2, 0), (2, 1)])],
                                [SingleNeuron([0, 1]), SingleNeuron([0, 1])]])

        self.X = np.array([[1, 3],
                           [2, 4]])

        self.y = np.array([3, 2])

    def test_get_differentials(self):
        derivatives = self.network.get_derivatives()
        self.network.update_network(self.X)

        # Calculate derivative numerically using taylor series
        dw = 1E-8
        for test_i, layer in enumerate(self.network.data):
            for test_j in range(len(layer)):
                analytical = derivatives[test_i][test_j](self.y)
                for k in range(self.network.data[test_i][test_j].w.shape[0]):
                    self.network.data[test_i][test_j].w[k] -= dw
                    output1 = self.network.update_network(self.X)
                    self.network.data[test_i][test_j].w[k] += 2*dw
                    output2 = self.network.update_network(self.X)
                    numerical = (calculate_log_likelihood(self.y, output2) - calculate_log_likelihood(self.y, output1))/(2*dw)
                    self.assertAlmostEqual(numerical, analytical[k], 5)

    def test_get_inputs(self):
        inputs = self.network.data[2][0].get_inputs(self.network.data, self.X)
        self.assertTrue((inputs[0] == np.array([[1],
                                                [2]])).all())
        self.assertTrue((inputs[1] == np.array([[3],
                                                [4]])).all())
        for index, neuron in enumerate(self.network.data[2]):
            neuron.output = index * np.array([1, 3])
        inputs = self.network.data[1][0].get_inputs(self.network.data, self.X)
        self.assertTrue((inputs[0] == np.array([[0],
                                                [0]])).all())
        self.assertTrue((inputs[1] == np.array([[1],
                                                [3]])).all())

    def test_update_X_tilde(self):
        neuron = self.network.data[2][0]
        neuron.update_X_tilde(self.network, self.X)
        self.assertTrue((neuron.X_tilde == np.array([[1, 1, 3],
                                                     [1, 2, 4]])).all())

    def test_update_output(self):
        neuron = self.network.data[2][0]
        neuron.update_X_tilde(self.network, self.X)
        neuron.update_output()
        self.assertTrue((neuron.output == predict(
            neuron.X_tilde, neuron.w)).all())
        self.assertEqual(neuron.output.shape, (2, ))

    def test_output(self):
        self.network.update_network(self.X)
        self.assertEqual(type(self.network.output()), np.ndarray)


def calculate_log_likelihood(y, output):
    L = 0
    for y, probability in zip(y, output):
        L += y * np.log(probability) + (1 - y) * np.log(1.0 - probability)
    return L


if __name__ == '__main__':
    unittest.main()
