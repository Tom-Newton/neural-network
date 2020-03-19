import unittest
import numpy as np
from network import Network
from single_neuron import SingleNeuron, Softmax, predict, logistic, softmax_predict


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.network = Network([[SingleNeuron([(1, 0), (1, 1)])],
                                [SingleNeuron([(2, 0), (2, 1)]),
                                 SingleNeuron([(2, 0), (2, 1)])],
                                [SingleNeuron([0, 1]), SingleNeuron([0, 1])]])

        self.X = np.array([[1, 3, 2],
                           [2, 4, 6]])

        self.y = np.array([3, 2])

        self.beta = np.array([2, 5, 8, 2, 9, 4, 5, 3, 7, 3, 8, 2, 11, 24, 6])

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
                    output1 = self.network.update_network(self.X)[0]
                    self.network.data[test_i][test_j].w[k] += 2*dw
                    output2 = self.network.update_network(self.X)[0]
                    numerical = (calculate_log_likelihood(
                        self.y, output2) - calculate_log_likelihood(self.y, output1))/(2*dw)
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
            neuron.X_tilde, neuron.w)[0]).all())
        self.assertEqual(neuron.output.shape, (2, ))

    def test_output(self):
        self.network.update_network(self.X)
        self.assertEqual(type(self.network.output()[0]), np.ndarray)

    def test_unpack_beta(self):
        self.network.unpack_beta(self.beta)
        self.assertListEqual(list(self.network.data[0][0].w), [2, 5, 8])
        self.assertListEqual(list(self.network.data[1][0].w), [2, 9, 4])
        self.assertListEqual(list(self.network.data[1][1].w), [5, 3, 7])
        self.assertListEqual(list(self.network.data[2][0].w), [3, 8, 2])
        self.assertListEqual(list(self.network.data[2][1].w), [11, 24, 6])

    def test_pack_beta(self):
        self.assertEqual(self.network.pack_beta().shape, (15,))

    def test_train(self):
        self.network.train(self.X, self.y, 1)


class SoftmaxTests(unittest.TestCase):
    def setUp(self):
        self.X_tilde = np.array([[1, 3, 2],
                                 [1, 4, 6],
                                 [1, 2, 3]])

        self.W = np.array([[2, 0.6],
                           [2.1, 1.9],
                           [1, 1.2]])

        self.number_classes = 2

        self.softmax = Softmax(self.number_classes, [])
        self.softmax.X_tilde = self.X_tilde
        self.softmax.W = self.W

    def test_softmax_predict(self):
        self.softmax.update_output()
        self.assertEqual(self.softmax.output.shape, (3, 2))


def calculate_log_likelihood(y, output):
    L = 0
    for y, probability in zip(y, output):
        L += y * np.log(probability) + (1 - y) * np.log(1.0 - probability)
    return L


if __name__ == '__main__':
    unittest.main()
