import unittest
import numpy as np
from network import Network
from single_neuron import SingleNeuron, predict


class Tests(unittest.TestCase):
    def setUp(self):
        self.network = Network([[SingleNeuron([(1, 0), (1, 1)])],
                                [SingleNeuron([(2, 0), (2, 1)]),
                                 SingleNeuron([(2, 0), (2, 1)])],
                                [SingleNeuron([0, 1]), SingleNeuron([0, 1])]])

        self.X = np.array([[1, 3],
                           [2, 4]])

    def test_get_differentials(self):
        self.network.get_derivatives()

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


if __name__ == '__main__':
    unittest.main()
