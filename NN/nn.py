import numpy as np
from NN import utilities as ut
from NN import forward_propagation as fp
from NN import backward_propagation as bp
from sklearn.preprocessing import normalize


def get_input():
    x, y = [], []
    with open('train.txt', 'r') as infile:
        m, l, n = infile.readline().split()
        m, l, n = int(m), int(l), int(n)
        d = (m, l, n)
        data_num = int(infile.readline())
        for i in range(data_num):
            values = infile.readline().split()
            values = list(map(float, values))
            x.append(values[:m]), y.append(values[m:])
    return x, y, d


def nn_model(x, y, d, iterations=10000, print_cost=True):
    grads = {}
    parameters = ut.initialize_parameters(d)
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    for i in range(iterations):

        a1, cache1 = fp.forward_propagation(x, w1, b1, sigmoid=False)
        a2, cache2 = fp.forward_propagation(a1, w2, b2, sigmoid=False)
        cost = ut.compute_cost(a2, y)
        da2 = (a2 - y) / (y.shape[1])
        da1, dw2, db2 = bp.backward_propagation(da2, cache2, sigmoid=False)
        _, dw1, db1 = bp.backward_propagation(da1, cache1, sigmoid=False)
        grads['dw1'] = dw1
        grads['db1'] = db1
        grads['dw2'] = dw2
        grads['db2'] = db2
        parameters = ut.update_parameters(parameters, grads)

        if i == 1:
            print("cost before training: %f" % cost)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


def main():
    x, y, d = get_input()
    x, y = np.array(x), np.array(y)
    x, y = x.T, y.T
    # print(x.shape)
    # print(y.shape)
    x = normalize(x, axis=0)
    parameters = nn_model(x, y, d, print_cost=False)
    a1, cache1 = fp.forward_propagation(x, parameters['w1'], parameters['b1'], sigmoid=False)
    a2, cache2 = fp.forward_propagation(a1, parameters['w2'], parameters['b2'], sigmoid=False)
    print("cost after training: " + str(ut.compute_cost(a2, y)))
    # print(x_test.shape)
    # print(x.shape)
    # print(y.shape)
    # print(d)


if __name__ == '__main__':
    main()
