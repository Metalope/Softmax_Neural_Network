from paramInitializer import initialize_parameters
import numpy as np

"""
This Class implements all functions to be executed by a linear layer
in a computational graph
"""


class LinearLayer:

    def __init__(self, input_shape, n_out, ini_type='plain', reg_param=None):
        """

        :param input_shape: input shape of Data/Activations
        :param n_out: number of neurons in layer
        :param ini_type: initialization type for weight parameters
        """
        self.m = input_shape[1]
        self.params = dict()
        self.params = initialize_parameters(input_shape[0], n_out, ini_type)
        self.Z = np.zeros((self.params['W'].shape[0], input_shape[1]))

        if reg_param is not None:
            self.reg_param = reg_param


    def linearForward(self, A_prev):
        """

        :param A_prev: Activations coming into the layer from previous layer
        """
        self.A_prev = A_prev
        self.Z = np.dot(self.params['W'], self.A_prev) + self.params['b']

    def linearBackward(self, upstream_grad, regularized=False):
        """

        :param upstream_grad: gradient coming in from the upper layer to couple with local gradient
        """
        self.dW = (1./self.m) * (np.dot(upstream_grad, self.A_prev.T))
        self.db = (1./self.m) * np.sum(upstream_grad, axis=1, keepdims=True)
        self.dA_prev = np.dot(self.params['W'].T, upstream_grad)

        if regularized:
            self.dW += (self.reg_param/self.m) * self.params['W']

    # add update params def
    def update_params(self, learning_rate):
        """

        :param learning_rate: learning rate hyper param for gradient descent
        """
        self.params['W'] = self.params['W'] - learning_rate * self.dW
        self.params['b'] = self.params['b'] - learning_rate * self.db

    def set_reg_param(self, reg_param):
        self.reg_param = reg_param

    def get_l2(self):
        return np.sum(np.square(self.params['W']))
