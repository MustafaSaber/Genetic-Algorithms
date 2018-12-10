import numpy as np


def forward_propagation(x, parameters):

    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    # x = (LEN_OF_SAMPLES, x_l)
    # y = (LEN_OF_SAMPLES, y_l)
    # w1 = (h_l, x_l)
    # b1 = (h_l, 1)
    # w2 = (y_l, h_l)
    # b2 = (y_l, 1)

    z1 = np.dot(w1, x.T) + b1
    a1 = z1
    z2 = np.dot(w2, a1) + b2
    a2 = z2

    # print("z1 = " + str(z1.shape))
    # print("z2 = " + str(z2.shape))
    # z1 =  (h_l, x_l) * (x_l , LEN_OF_SAMPLES) = (h_l , LEN_OF_SAMPLES) = (10, 515)
    # a1 = (h_l , LEN_OF_SAMPLES)
    # z2 = (y_l, h_l) * (h_l, LEN_OF_SAMPLES) =  (y_l, LEN_OF_SAMPLES) =  (1, 515)
    # a2 = (y_l, LEN_OF_SAMPLES)

    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

    return a2, cache
