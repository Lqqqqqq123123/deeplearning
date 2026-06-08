import numpy as np
from numpy import array

def step_function(x):
    """
    阶跃激活函数
    :param x: ndarray
    :return: ndarray
    """
    return np.array(x > 0, dtype=np.int32)


def sigmoid(x):

    return 1 / (1 + np.exp(-x))


"""
    Tanh函数
    Args:
        x (ndarray): 输入数据
    Returns:
        ndarray: Tanh函数输出
"""


def tanh(x):
    return np.tanh(x)


def reLu(x):
    return np.maximum(0, x)


def leakReLu(x, a=0.01):
    if x > 0:
        return x
    else:
        return a * x


def softmax(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    else:
        return np.exp(x - np.max(x)) / np.sum(np.exp(x))

def softmax_2(x):
    # x:(2, 3)
    if x.ndim == 2:
        x = x.T  # (3, 2)
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def identity(x):
    return x


