import numpy as np
from NN import utilities as ut
from NN import forward_propagation as fp
from NN import backward_propagation as bp


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


def nn_model(x, y, d, iterations=1000, print_cost=True):

    parameters = ut.initialize_parameters(d)

    for i in range(iterations):
        a2, cache = fp.forward_propagation(x, parameters)
        # print(cache)
        cost = ut.compute_cost(a2, y)
        grads = bp.backward_propagation(parameters, cache, x, y)
        parameters = ut.update_parameters(parameters, grads)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


def main():
    x, y, d = get_input()
    x, y = np.array(x), np.array(y)
    # x, y = np.squeeze(x.T), np.squeeze(y.T)
    x, y = x.T, y.T
    parameters = nn_model(x, y, d)
    a2, _ = fp.forward_propagation(x, parameters)
    print(ut.compute_cost(a2, y))
    # print(x.shape)
    # print(y.shape)
    # print(d)


if __name__ == '__main__':
    main()
