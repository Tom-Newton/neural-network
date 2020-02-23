import unittest
import network
from single_neuron import SingleNeuron


class Tests(unittest.TestCase):
    def setUp(self):
        self.network = [[SingleNeuron([(1, 0), (1, 1)])],
                        [SingleNeuron([(2, 0), (2, 1)]), SingleNeuron([(2, 0), (2, 1)])],
                        [SingleNeuron([0, 1]), SingleNeuron([0, 1])]]

    def test_get_differentials(self):
        differentials = network.get_derivatives(self.network)
        print(differentials)

if __name__ == '__main__':
    unittest.main()