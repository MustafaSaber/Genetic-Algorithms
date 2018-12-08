import numpy as np


def backward_propagation(parameters, cache, x, y):
    m = x.shape[1]

    w1 = parameters['w1']
    w2 = parameters['w2']

    a1 = cache['a1']
    a2 = cache['a2']

    dz2 = a2 - y
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.dot(w2.T, dz2) * (a1 * (1 - a1))
    dw1 = (1 / m) * np.dot(dz1, x.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}

    return grads
