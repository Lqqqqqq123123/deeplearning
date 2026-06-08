import numpy as np
from numpy import ndarray


def sigmoid(x:ndarray):
    """
    :param x: 可以是标量,也可以是ndarray
    :return:
    """
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x


def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])

if __name__ == '__main__':
    print(sigmoid(np.array([1, 2, 3])))