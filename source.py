import numpy as np
import matplotlib.pyplot as plt
from utilities import *
import SoftmaxLayer
import LinearLayer


np.set_printoptions(precision=3, suppress=True)


X, y, colormap = loadData() # X->(20,2) ; y->(20,)

plt.figure(1)
plot_scatter(X, y, colormap)

# one hot encoded matrix
Y = one_hot_encode(y, num_classes=3) # Y-> (20,3)

train_X = X.T  # train_X -> (2, 20)
train_Y = Y.T  # train_Y -> (3, 20)

# --------------------------------- MODEL ---------------------------------

learning_rate = 2.0
np.random.seed(0)
num_of_epochs = 100
costs =[] # list to store the costs over the training period


Z1 = LinearLayer.LinearLayer(input_shape=train_X.shape, n_out=3)
A1 = SoftmaxLayer.SoftmaxLayer(Z1.Z.shape)

for epoch in range(num_of_epochs):

    # ---------------------- forward propagation ---------------------------

    Z1.linearForward(train_X)
    A1.activate(Z1.Z)

    # compute cost
    #cost = compute_softmax_cost(A1.A, train_Y)
    cost = compute_stable_softmax_cost(Z1.Z, train_Y)
    if (epoch % 1) == 0:
        print("Cost at epoch#" + str(epoch) + ": " + str(cost))
        costs.append(cost)

    # ----------------------- back propagation -----------------------------
    A1.backward(labels=train_Y)
    Z1.linearBackward(upstream_grad=A1.dZ)

    # ----------------------- back propagation -----------------------------
    Z1.update_params(learning_rate)



# plot the cost
plt.figure(2)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations ')
plt.title("Learning rate =" + str(learning_rate))
plt.show()

plt.figure(3)
plt.style.use('classic')
axes = plt.gca()
plt.xlim([-2.0, 2.0])
plt.ylim([-2.0, 2.0])
plt.xlabel('$x_1$', size=20)
plt.ylabel('$x_2$', size=20)
plt.title('Decision boundary', size=18)

plot_decision_boundary(lambda x: predict_dec(Zs=[Z1],As=[A1], X=x.T), train_X.T, y)
