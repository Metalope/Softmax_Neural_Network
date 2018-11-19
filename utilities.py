import numpy as np
import matplotlib.pyplot as plt


def compute_softmax_cost(AL, Y):
    """

    :param AL: Activations from the last layer
    :param Y: labels of data
    :return:
        - cost: The cross-entropy cost function(logistic cost function) result
    """
    m = Y.shape[1]


    cost = (1/m) * np.sum(np.multiply(-Y, np.log(AL))) # vanilla version

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost


def compute_stable_softmax_cost(ZL, Y):
    """
    Computes numerically stable Softmax cost
    http://saitcelebi.com/tut/output/part2.html#numerical_stability_of_softmax_function

    :param ZL: Linear values from the last layer
    :param Y: labels of data
    :return:
        - cost: The cross-entropy cost function(logistic cost function) result
    """
    m = Y.shape[1]

    max_val = np.amax(ZL, axis=0, keepdims=True)
    cost = (1/m) * np.sum( np.multiply(-Y, np.subtract(ZL - max_val, np.log(np.sum(np.exp(ZL - max_val), axis=0, keepdims=True)))))

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost

def loadData():
    X = np.array([[-0.1, 1.4],
                  [-0.5, 0.2],
                  [1.3, 0.9],
                  [-0.6, 0.4],
                  [-1.6, 0.2],
                  [0.2, 0.2],
                  [-0.3, -0.4],
                  [0.7, -0.8],
                  [1.1, -1.5],
                  [-1.0, 0.9],
                  [-0.5, 1.5],
                  [-1.3, -0.4],
                  [-1.4, -1.2],
                  [-0.9, -0.7],
                  [0.4, -1.3],
                  [-0.4, 0.6],
                  [0.3, -0.5],
                  [-1.6, -0.7],
                  [-0.5, -1.4],
                  [-1.0, -1.4]])

    y = np.array([0, 0, 1, 0, 2, 1, 1, 1, 1, 0, 0, 2, 2, 2, 1, 0, 1, 2, 2, 2])


    colormap = np.array(['r', 'g', 'b'])

    return X,y, colormap

def plot_scatter(X, y, colormap):
    plt.style.use('classic')
    plt.grid()
    plt.xlim([-2.0, 2.0])
    plt.ylim([-2.0, 2.0])
    plt.xlabel('$x_1$', size=20)
    plt.ylabel('$X_2$', size=20)
    plt.title('Input 2D points', size=18)
    plt.scatter(x=X[:, 0], y=X[:, 1], s=50, c=colormap[y])


def one_hot_encode(y,num_classes):
    return np.eye(num_classes)[y]


def predict_dec(Zs, As, X):
    """
    Used for plotting decision boundary.

    Arguments:
    Zs -- linear layers
    As -- Activation layers
    X -- input data of size (m, K)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Predict using forward propagation and
    m = X.shape[1]
    n = len(Zs)  # number of layers in the neural network

    # Forward propagation
    Zs[0].linearForward(X)
    As[0].activate(Zs[0].Z)
    for i in range(1, n):
        Zs[i].linearForward(As[i - 1].A)
        As[i].activate(Zs[i].Z)
    probas = As[n - 1].A

    # class with the highest probability gets selected
    predictions = np.argmax(probas.T, axis=1)

    return predictions

def plot_decision_boundary(model, X, y):

    # Generate a grid of points between -2.0 and 2.0 with 1000 points in between
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xs = np.linspace(-2.0, 2.0, 1000)
    ys = np.linspace(2.0, -2.0, 1000)
    xx, yy = np.meshgrid(xs, ys)
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples

    cmap = plt.cm.get_cmap("Spectral")

    plt.contourf(xx, yy, Z, cmap=cmap)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], s=50, c=np.squeeze(y), cmap=cmap)
    plt.show()

