import numpy as np


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    cache = z
    return a, cache


def sigmoid_backward(da, cache):
    z = cache
    s = 1 / (1 + np.exp(-z))
    dz = da * s * (1 - s)
    return dz


def initialize_parameters(d):
    x, h, y = d
    w1 = np.random.randn(h, x) * 0.01
    # w1 = (h_l, x_l)
    b1 = np.zeros((h, 1))
    # b1 = (h_l, 1)
    w2 = np.random.randn(y, h) * 0.01
    # w2 = (y_l, h_l)
    b2 = np.zeros((y, 1))
    # b2 = (y_l, 1)

    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return parameters


def compute_cost(a2, y):
    m = y.shape[1]
    e = a2 - y
    se = np.dot(e, e.T)
    mse = np.sum(se) / m
    mse = np.squeeze(mse)
    return mse


def update_parameters(parameters, grads, lr=1.3):

    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']
    # print(b2.shape)
    # print(db2.shape)
    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2

    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return parameters

