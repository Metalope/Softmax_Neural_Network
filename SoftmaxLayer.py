import numpy as np

class SoftmaxLayer:
    def __init__(self, shape):
        self.A = np.zeros(shape)


    def activate(self, Z):
        # http: // saitcelebi.com / tut / output / part2.html  # numerical_stability_of_softmax_function
        max_val = np.amax(Z, axis=0, keepdims=True)  # To normalize the values for numerical stability
        self.A = np.exp(Z - max_val)/np.sum(np.exp(Z - max_val), axis=0, keepdims=True)

    def backward(self, labels):
        # ones = np.ones(self.A.shape)
        # eye = np.eye(self.A.shape[0])
        # local_grad = self.A.dot(ones.T) * (eye - ones.dot(self.A.T) )
        # self.dZ = upstream_grad * local_grad

        self.dZ = (self.A - labels) / labels.shape[-1]  # divide by size of batch

