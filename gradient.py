import util
from neuralnet import *
import copy
import numpy as np

def check_grad(model, x_train, y_train, epsilon, index):

    """
    TODO
        Checks if gradients computed numerically are within O(epsilon**2)

        args:
            model
            x_train: Small subset of the original train dataset
            y_train: Corresponding target labels of x_train

        Prints gradient difference of values calculated via numerical approximation and backprop implementation
    """
    i, j, k = index
    model_1 = copy.deepcopy(model)
    model_2 = copy.deepcopy(model)

    model.forward(x_train)
    model.backward(gradReqd = False, targets = y_train)
    gradients = []
    for a in range(model.num_layers):
        gradients.append(model.layers[a].gradient)
    gradient = gradients[i][j][k]

    model_1.epsilon_change(epsilon, index)
    model_2.epsilon_change(epsilon, index, add = False)

    loss_1 = model_1.loss(model_1.forward(x_train), targets = y_train) * y_train.shape[0]
    loss_2 = model_2.loss(model_2.forward(x_train), targets = y_train) * y_train.shape[0]
    numapprx = (loss_1 - loss_2)/ (2 * epsilon)
    diff = np.abs(numapprx - gradients[i][j][k])
    return gradient, numapprx, diff
    

def checkGradient(x_train, y_train, config):

    subsetSize = 10  #Feel free to change this
    sample_idx = np.random.randint(0,len(x_train), subsetSize)
    x_train_sample, y_train_sample = x_train[sample_idx], y_train[sample_idx]
    epsilon = config["Epsilon"]
    index = tuple(config["Index"])
    i, j, k = index

    model = Neuralnetwork(config)
    gradient, numapprx, diff = check_grad(model, x_train_sample, y_train_sample, epsilon, index)

    if i == 0:
        if j == 0:
            print('Weight Type: Hidden bias', 'Gradient:', gradient, 'Numerical Approximation:', numapprx, 'Difference:', diff)
        else:
            print('Weight Type: Input to hidden', 'Gradient:', gradient, 'Numerical Approximation:', numapprx, 'Difference:', diff)
    elif i == (len(config['layer_specs']) - 2):
        if j == 0:
            print('Weight Type: Output bias', 'Gradient:', gradient, 'Numerical Approximation:', numapprx, 'Difference:', diff)
        else:
            print('Weight Type: Hidden to output', 'Gradient:', gradient, 'Numerical Approximation:', numapprx, 'Difference:', diff)