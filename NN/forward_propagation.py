import numpy as np
from NN import utilities as ut


def forward(a, w, b):
    z = np.dot(w, a) + b
    cache = (a, w, b)
    return z, cache


def forward_propagation(prev, w, b, sigmoid=True):

    # x = (LEN_OF_SAMPLES, x_l)
    # y = (LEN_OF_SAMPLES, y_l)
    # w1 = (h_l, x_l)
    # b1 = (h_l, 1)
    # w2 = (y_l, h_l)
    # b2 = (y_l, 1)

    z, linear_cache = forward(prev, w, b)
    if sigmoid:
        a, activation_cache = ut.sigmoid(z)
        cache = (linear_cache, activation_cache)
    else:
        return z, linear_cache

    # z1 =  (h_l, x_l) * (x_l , LEN_OF_SAMPLES) = (h_l , LEN_OF_SAMPLES) = (10, 515)
    # a1 = (h_l , LEN_OF_SAMPLES)
    # z2 = (y_l, h_l) * (h_l, LEN_OF_SAMPLES) =  (y_l, LEN_OF_SAMPLES) =  (1, 515)
    # a2 = (y_l, LEN_OF_SAMPLES)

    return a, cache
