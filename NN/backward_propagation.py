import numpy as np


def backward_propagation(parameters, cache, x, y):
    m = x.shape[0]

    w2 = parameters['w2']

    a1 = cache['a1']
    a2 = cache['a2']

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

    dz2 = a2 - y.T
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.dot(w2.T, dz2)
    dw1 = (1 / m) * np.dot(dz1, x)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    # print("x = " + str(x.shape))
    # print("y = " + str(y.shape))
    # print("dz1 = " + str(dz1.shape))
    # print("dz2 = " + str(dz2.shape))
    # print("dw1 = " + str(dw1.shape))
    # print("dw2 = " + str(dw2.shape))
    # print("db1 = " + str(db1.shape))
    # print("db2 = " + str(db2.shape))
    # dz2 = (y_l, LEN_OF_SAMPLES) - (y_l, LEN_OF_SAMPLES) = (y_l, LEN_OF_SAMPLES) = (1, 515)
    # dw2 =  (y_l, LEN_OF_SAMPLES) * (LEN_OF_SAMPLES, h_l) =  (y_l, h_l) = (1, 10)
    # db2 = (y_l, LEN_OF_SAMPLES)  = (y_l, 1)
    # dz1 = (h_l, y_l) * (y_l, LEN_OF_SAMPLES) = (h_l, LEN_OF_SAMPLES) = (10, 515)
    # dw1 = (h_l, LEN_OF_SAMPLES) * (LEN_OF_SAMPLES, x_l) = (h_l, x_l)
    # db1 = (h_l, LEN_OF_SAMPLES) = (h_1, 1)

    grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}

    return grads
