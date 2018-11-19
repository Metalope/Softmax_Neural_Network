import numpy as np


def initialize_parameters(n_in, n_out, ini_type='plain'):
    """

    :param n_in: size of input layer
    :param n_out: size of output/number of neurons
    :param ini_type: set initialization type for weights
    :return: "params" a dictionary containing W and b
    """

    params = dict()  # initialize empty dictionary

    if ini_type == 'plain':
        params['W'] = np.random.randn(n_out, n_in) * 0.01  # set weights 'W' to small random gaussian
    elif ini_type == 'xavier':
        params['W'] = np.random.randn(n_out, n_in) / (np.sqrt(n_in))  # set variance of W to 1/n
    elif ini_type == 'he':
        # Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
        # Kaiming He et al. (https://arxiv.org/abs/1502.01852)
        # http: // cs231n.github.io / neural - networks - 2 /  # init
        params['W'] = np.random.randn(n_out, n_in) * np.sqrt(2/n_in)  # set variance of W to 2/n

    params['b'] = np.zeros((n_out, 1))    # set bias 'b' to zeros
    # params['W'] = np.array([[0.31, 3.95],
    #                         [7.07, -0.23],
    #                         [-6.27, -2.35]])
    # params['b'] = np.array([[1.2],
    #                         [2.93],
    #                         [-4.14]])


    return params


