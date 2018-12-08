import numpy as np


def forward_propagation(x, parameters):

    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    z1 = np.dot(w1, x) + b1
    a1 = 1. / (1. + np.exp(-z1))
    z2 = np.dot(w2, a1) + b2
    a2 = 1. / (1. + np.exp(-z2))

    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

    return a2, cache
