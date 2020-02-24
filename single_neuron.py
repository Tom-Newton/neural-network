import numpy as np

# inputs is a list containing tuples:
# If the input_type is a neuron_output (i, j)
# If the input_type isa data_input n
class SingleNeuron:
    def __init__(self, inputs):
        self.inputs = inputs
        self.w = np.random.randn(len(inputs) + 1)
        self.X_tilde = None

    def output(self):
        return predict(self.X_tilde, self.w)

    
def logistic(x): return 1.0 / (1.0 + np.exp(-x))

def predict(X_tilde, w): return logistic(np.dot(X_tilde, w))