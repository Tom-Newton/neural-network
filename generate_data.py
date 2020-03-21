import numpy as np
import matplotlib.pyplot as plt

number_samples = 1000
number_input_dimensions = 2
index = 0

X = np.zeros((number_samples, number_input_dimensions))
y = np.zeros((number_samples,))


def normal_samples(mean, standard_deviation, dimension, number_samples):
    return mean + standard_deviation*np.random.randn(number_samples, dimension)


def add_samples(n, category, mean, standard_deviation):
    global index
    X[index: index + n,
        :] = normal_samples(mean, standard_deviation, number_input_dimensions, n)
    y[index: index + n] += category
    index += n


add_samples(200, 1, np.array([0, 2]), 0.5)
add_samples(200, 0, np.array([0, 0]), 0.5)
add_samples(200, 1, np.array([3, 2]), 0.5)
add_samples(200, 2, np.array([0, -2]), 0.5)
add_samples(100, 2, np.array([1.5, 1.5]), 0.8)
add_samples(100, 0, np.array([2, 0]), 0.5)


with open('X2.np','wb') as f:
    np.save(f, X)

mat = np.matrix(y)
with open('y2.np','wb') as f:
    np.save(f, y)

plt.figure()
plt.plot(X[y == 0, 0], X[y == 0, 1], 'bo')
plt.plot(X[y == 1, 0], X[y == 1, 1], 'ro')
plt.plot(X[y == 2, 0], X[y == 2, 1], 'go')
plt.show()
