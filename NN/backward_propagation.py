import numpy as np
from NN import utilities as ut


def backward(dz, cache):
    a_prev, w, b = cache
    m = a_prev.shape[1]

    dw = np.dot(dz, a_prev.T) / m
    db = np.sum(dz, axis=1, keepdims=True) / m
    da_prev = np.dot(w.T, dz)

    return da_prev, dw, db


def backward_propagation(da, cache, sigmoid=True):

    # x = (LEN_OF_SAMPLES, x_l)
    # y = (LEN_OF_SAMPLES, y_l)
    # w1 = (h_l, x_l)
    # b1 = (h_l, 1)
    # w2 = (y_l, h_l)
    # b2 = (y_l, 1)

    # z1 =  (h_l, x_l) * (x_l , LEN_OF_SAMPLES) = (h_l , LEN_OF_SAMPLES) = (10, 515)
    # a1 = (h_l , LEN_OF_SAMPLES)
    # z2 = (y_l, h_l) * (h_l, LEN_OF_SAMPLES) =  (y_l, LEN_OF_SAMPLES) =  (1, 515)
    # a2 = (y_l, LEN_OF_SAMPLES)
    if sigmoid:
        linear_cache, activation_cache = cache
        dz = ut.sigmoid_backward(da, activation_cache)
        da_prev, dw, db = backward(dz, linear_cache)
    else:
        linear_cache = cache
        da_prev, dw, db = backward(da, linear_cache)

    # dz2 = (y_l, LEN_OF_SAMPLES) - (y_l, LEN_OF_SAMPLES) = (y_l, LEN_OF_SAMPLES) = (1, 515)
    # dw2 =  (y_l, LEN_OF_SAMPLES) * (LEN_OF_SAMPLES, h_l) =  (y_l, h_l) = (1, 10)
    # db2 = (y_l, LEN_OF_SAMPLES)  = (y_l, 1)
    # dz1 = (h_l, y_l) * (y_l, LEN_OF_SAMPLES) = (h_l, LEN_OF_SAMPLES) = (10, 515)
    # dw1 = (h_l, LEN_OF_SAMPLES) * (LEN_OF_SAMPLES, x_l) = (h_l, x_l)
    # db1 = (h_l, LEN_OF_SAMPLES) = (h_1, 1)

    return da_prev, dw, db
