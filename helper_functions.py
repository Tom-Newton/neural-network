import numpy as np
import pickle

def transform_image(image):
    x = np.array([])
    for row in image:
        x = np.concatenate((x, row), axis=0)
    return x


def un_transform_image(x, shape):
    image = np.zeros(shape)
    index = 0
    for i in range(shape[0]):
        image[i, :] = x[index: index + shape[1]]
        index += shape[1]
    return image


def normalise(x):
    n_data = x.shape[0]
    m = x.sum(axis=0)
    x0 = x - m/n_data
    s = np.sqrt((x0**2).sum(axis=0)/n_data)
    ss = np.array([tmp if tmp != 0 else 1 for tmp in s])
    x00 = x0 / ss
    return x00

def unpickle(file):
    with open (file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data