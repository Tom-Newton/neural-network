import unittest
import unittest.mock
import numpy as np
import cupy as cp
from network import Network, log_likelihood, calculate_confusion_matrix
from single_neuron import SingleNeuron, Softmax, Convolutional, predict, logistic, softmax_predict


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.network = Network([[Softmax(2, [(1, 0), (1, 1), 0])],
                                [SingleNeuron([(2, 0), (2, 1)]),
                                 SingleNeuron([(2, 1)])],
                                [SingleNeuron([0, 1]), SingleNeuron([0, 1])]])

        self.w_data = [
            [cp.array([[1, 3, 0, 6],
                       [1, 2, -1, 3]])],
            [cp.array([2, -1, 0]), cp.array([-1, 2, 2])],
            [cp.array([2, -1, 0]), cp.array([-1, 2, 2])]
        ]

        self.X = cp.array([[1, 3, 2],
                           [2, 4, 6]])

        self.Y = cp.array([[0, 1],
                           [1, 0]])

        self.beta = cp.array(
            [2, 5, 8, 1, 2, 5, 7, 1, 2, 9, 4, 5, 3, 7, 3, 8, 2, 11, 24, 6])

    def test_log_likelihood(self):
        output = self.network.update_network(self.X)[0]
        self.assertEqual(log_likelihood(self.Y, output).shape, ())

    def test_get_differentials(self):
        derivatives = self.network.get_derivatives()
        self.network.update_network(self.X)

        # Calculate derivative numerically using taylor series
        dw = 1E-6
        for test_i, layer in enumerate(self.network.data):
            for test_j in range(len(layer)):
                analytical = derivatives[test_i][test_j](self.Y)
                for k in range(self.network.data[test_i][test_j].w.shape[0]):
                    if type(self.network.data[test_i][test_j]) == Softmax:
                        for l in range(self.network.data[0][0].number_classes):
                            self.network.data[0][0].W[k][l] -= dw
                            output1 = self.network.update_network(self.X)[0]
                            self.network.data[0][0].W[k][l] += 2*dw
                            output2 = self.network.update_network(self.X)[0]
                            numerical = (log_likelihood(
                                self.Y, output2) - log_likelihood(self.Y, output1))/(2*dw)
                            self.assertAlmostEqual(
                                cp.asnumpy(numerical), cp.asnumpy(analytical[k][l]), 5)
                    else:
                        self.network.data[test_i][test_j].w[k] -= dw
                        output1 = self.network.update_network(self.X)[0]
                        self.network.data[test_i][test_j].w[k] += 2*dw
                        output2 = self.network.update_network(self.X)[0]
                        numerical = (log_likelihood(
                            self.Y, output2) - log_likelihood(self.Y, output1))/(2*dw)
                        self.assertAlmostEqual(cp.asnumpy(
                            numerical), cp.asnumpy(analytical[k]), 5)

    def test_get_inputs(self):
        inputs = self.network.data[2][0]._get_inputs(self.network.data, self.X)
        self.assertTrue((inputs[0] == cp.array([[1],
                                                [2]])).all())
        self.assertTrue((inputs[1] == cp.array([[3],
                                                [4]])).all())
        for index, neuron in enumerate(self.network.data[2]):
            neuron.output = index * cp.array([1, 3])
        inputs = self.network.data[1][0]._get_inputs(self.network.data, self.X)
        self.assertTrue((inputs[0] == cp.array([[0],
                                                [0]])).all())
        self.assertTrue((inputs[1] == cp.array([[1],
                                                [3]])).all())

    def test_update_X_tilde(self):
        neuron = self.network.data[2][0]
        neuron._update_X_tilde(self.network, self.X)
        self.assertTrue((neuron.X_tilde == cp.array([[1, 1, 3],
                                                     [1, 2, 4]])).all())

    def test_update_output(self):
        neuron = self.network.data[2][0]
        neuron.update_output(self.network, self.X)
        self.assertTrue((neuron.get_output() == predict(
            neuron.X_tilde, neuron.w)[0]).all())
        self.assertEqual(neuron.get_output().shape, (2, ))

    def test_output(self):
        self.network.update_network(self.X)
        self.assertEqual(type(self.network.get_output()[0]), cp.ndarray)

    def test_reset_weights(self):
        self.network.reset_weights()

    def test_set_weights(self):
        self.network.set_weights(self.w_data)
        for w_layer, layer in zip(self.w_data, self.network.data):
            for w, neuron in zip(w_layer, layer):
                if type(neuron) == Softmax:
                    self.assertEqual(w.shape, neuron.W.shape)
                else:
                    self.assertListEqual(list(w), list(neuron.w))

    def test_unpack_beta(self):
        self.network.unpack_beta(self.beta)
        self.assertListEqual(
            list(self.network.data[0][0].W[:, 0]), [2, 5, 8, 1])
        self.assertListEqual(
            list(self.network.data[0][0].W[:, 1]), [2, 5, 7, 1])
        self.assertListEqual(list(self.network.data[1][0].w), [2, 9, 4])
        self.assertListEqual(list(self.network.data[1][1].w), [5, 3, 7])
        self.assertListEqual(list(self.network.data[2][0].w), [3, 8, 2])
        self.assertListEqual(list(self.network.data[2][1].w), [11, 24, 6])

    def test_pack_beta(self):
        self.network.unpack_beta(self.beta)
        beta = self.network.pack_beta()
        self.assertListEqual(list(beta), list(self.beta))

    def test_train(self):
        self.network.train(self.X, self.Y, 1, 20)

    def test_calculate_confusion_matrix(self):
        calculate_confusion_matrix(self.Y, cp.array([[0.4, 0.6],
                                                     [0.4, 0.6]]))


class SoftmaxTests(unittest.TestCase):
    def setUp(self):
        self.X_tilde = cp.array([[1, 3, 2],
                                 [1, 4, 6],
                                 [1, 2, 3]])

        self.W = cp.array([[2, 0.6],
                           [2.1, 1.9],
                           [1, 1.2]])

        self.number_classes = 2

        self.softmax = Softmax(self.number_classes, [])
        self.softmax.X_tilde = self.X_tilde
        self.softmax.W = self.W

        self.softmax._update_X_tilde = unittest.mock.Mock()

    def test_softmax_predict(self):
        self.softmax.update_output(None, None)
        self.assertEqual(self.softmax.get_output().shape, (3, 2))


class ConvolutionalTests(unittest.TestCase):
    def setUp(self):
        self.w = cp.array([1, 2, 3, 4, 5, 6, 7])
        self.convolutional = Convolutional(
            input_location=0, filter_shape=(2, 3))
        self.convolutional.w = self.w

    def test_get_W_matrix(self):
        W, bias = self.convolutional.get_W_matrix()
        self.assertEqual(bias, 1)
        cp.testing.assert_array_equal(W, cp.array([[2, 3, 4],
                                                   [5, 6, 7]]))


class ConvolutionalNetworkTests(unittest.TestCase):
    def setUp(self):
        self.network = Network([[Softmax(number_classes=2, input_locations=[(1, 0), (1, 1), (1, 2)])],
                                [SingleNeuron(input_locations=[(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5)]),
                                 SingleNeuron(input_locations=[(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5)]),
                                 SingleNeuron(input_locations=[(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)])],
                                [Convolutional(input_location=(3, 0), filter_shape=(1, 2)),
                                 Convolutional(input_location=(3, 1), filter_shape=(1, 2)),
                                 Convolutional(input_location=(3, 2), filter_shape=(1, 2)),
                                 Convolutional(input_location=(3, 3), filter_shape=(1, 2)),
                                 Convolutional(input_location=(3, 4), filter_shape=(1, 2)),
                                 Convolutional(input_location=0, filter_shape=(5, 5))],
                                [Convolutional(input_location=(4, 0), filter_shape=(2, 2)),
                                 Convolutional(input_location=(4, 1), filter_shape=(2, 2)),
                                 Convolutional(input_location=(4, 2), filter_shape=(2, 2)),
                                 Convolutional(input_location=(4, 1), filter_shape=(2, 2)),
                                 Convolutional(input_location=(4, 2), filter_shape=(2, 2))],
                                [Convolutional(input_location=(5, 0), filter_shape=(2, 2)),
                                 Convolutional(input_location=(5, 0), filter_shape=(2, 2)),
                                 Convolutional(input_location=(5, 1), filter_shape=(2, 2))],
                                [Convolutional(input_location=0, filter_shape=(3, 2)),
                                 Convolutional(input_location=0, filter_shape=(3, 2))]])

        self.w_data = [
            [cp.array([[1, 3, 0],
                       [1, 2, -1]])],
            [cp.array([2, -1, 0, 5, 6])],
            [cp.array([2, -1, 0, 2, 7])]
        ]

        self.X = cp.array([
            [
                [[1, 2, 3, 4, 2],
                 [4, 5, 6, 5, 7],
                 [3, 2, 5, 1, 7],
                 [3, 7, 1, 9, 9],
                 [3, 7, 1, 9, 9]], ],
            [
                [[3, 1, 2, 5, 3],
                 [2, 6, 4, 7, 2],
                 [3, 7, 1, 9, 9],
                 [3, 7, 1, 9, 9],
                 [3, 7, 1, 9, 9]], ]
        ])

        self.Y = cp.array([[0, 1],
                           [1, 0]])

    def test_get_inputs(self):
        inputs = self.network.data[5][0]._get_inputs(self.network.data, self.X)
        cp.testing.assert_array_equal(inputs, [cp.array([
            [[1, 2, 3, 4, 2],
             [4, 5, 6, 5, 7],
             [3, 2, 5, 1, 7],
             [3, 7, 1, 9, 9],
             [3, 7, 1, 9, 9]],

            [[3, 1, 2, 5, 3],
             [2, 6, 4, 7, 2],
             [3, 7, 1, 9, 9],
             [3, 7, 1, 9, 9],
             [3, 7, 1, 9, 9]],
        ])])
        self.network.data[3][0].output_array = cp.array([[[1, 2]],
                                                         [[3, 1]], ])
        inputs = self.network.data[2][0]._get_inputs(self.network.data, self.X)
        cp.testing.assert_array_equal(inputs, [cp.array([
            [[1, 2]],
            [[3, 1]],
        ])])

    def test_update_X_tilde(self):
        neuron = self.network.data[5][0]
        neuron._update_X_tilde(self.network, self.X)
        cp.testing.assert_array_equal(neuron.get_X_tilde(a=0, b=0), cp.array([[1, 1, 2, 4, 5, 3, 2],
                                                                              [1, 3, 1, 2, 6, 3, 7]]))
        cp.testing.assert_array_equal(neuron.get_X_tilde(a=0, b=1), cp.array([[1, 2, 3, 5, 6, 2, 5],
                                                                              [1, 1, 2, 6, 4, 7, 1]]))

    def test_update_output(self):
        neuron = self.network.data[5][0]
        neuron.update_output(self.network, self.X)
        cp.testing.assert_array_equal(
            neuron.get_output(a=0, b=0), predict(neuron.get_X_tilde(a=0, b=0), neuron.w)[0])
        self.assertEqual(neuron.get_output(a=0, b=0).shape, (2,))
        self.assertEqual(neuron.get_output().shape, (2, 3, 4))

    def test_output(self):
        self.network.update_network(self.X)
        self.assertEqual(type(self.network.get_output()[0]), cp.ndarray)

    def test_reset_weights(self):
        self.network.reset_weights()

    def test_set_weights(self):
        self.network.set_weights(self.w_data)
        for w_layer, layer in zip(self.w_data, self.network.data):
            for w, neuron in zip(w_layer, layer):
                if type(neuron) == Softmax:
                    self.assertEqual(w.shape, neuron.W.shape)
                else:
                    self.assertListEqual(list(w), list(neuron.w))


    def test_get_differentials(self):
        derivatives = self.network.get_derivatives()
        self.network.update_network(self.X)

        # Calculate derivative numerically using taylor series
        dw = 1E-6
        for test_i, layer in enumerate(self.network.data):
            for test_j in range(len(layer)):
                analytical = derivatives[test_i][test_j](self.Y)
                for k in range(self.network.data[test_i][test_j].w.shape[0]):
                    if type(self.network.data[test_i][test_j]) == Softmax:
                        for l in range(self.network.data[0][0].number_classes):
                            self.network.data[0][0].W[k][l] -= dw
                            output1 = self.network.update_network(self.X)[0]
                            self.network.data[0][0].W[k][l] += 2*dw
                            output2 = self.network.update_network(self.X)[0]
                            numerical = (log_likelihood(
                                self.Y, output2) - log_likelihood(self.Y, output1))/(2*dw)
                            self.assertAlmostEqual(
                                cp.asnumpy(numerical), cp.asnumpy(analytical[k][l]), 5)
                    else:
                        self.network.data[test_i][test_j].w[k] -= dw
                        output1 = self.network.update_network(self.X)[0]
                        self.network.data[test_i][test_j].w[k] += 2*dw
                        output2 = self.network.update_network(self.X)[0]
                        numerical = (log_likelihood(
                            self.Y, output2) - log_likelihood(self.Y, output1))/(2*dw)
                        self.assertAlmostEqual(cp.asnumpy(
                            numerical), cp.asnumpy(analytical[k]), 5)
    
    def test_train(self):
        self.network.train(self.X, self.Y, 1, 20)


if __name__ == '__main__':
    unittest.main()
