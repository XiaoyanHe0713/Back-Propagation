import numpy as np
import util

class Activation():
    def __init__(self, activation_type = "sigmoid"):
        if activation_type not in ["sigmoid", "tanh", "ReLU", "output"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        self.activation_type = activation_type

    def __call__(self, z):
        return self.forward(z)

    def forward(self, z):
        if self.activation_type == "sigmoid":
            return self.sigmoid(z)

        elif self.activation_type == "tanh":
            return self.tanh(z)

        elif self.activation_type == "ReLU":
            return self.ReLU(z)

        elif self.activation_type == "output":
            return self.output(z)

    def backward(self, z):
        if self.activation_type == "sigmoid":
            return self.grad_sigmoid(z)

        elif self.activation_type == "tanh":
            return self.grad_tanh(z)

        elif self.activation_type == "ReLU":
            return self.grad_ReLU(z)

        elif self.activation_type == "output":
            return self.grad_output(z)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def ReLU(self, x):
        return np.maximum(0, x)

    def output(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def grad_sigmoid(self,x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def grad_tanh(self,x):
        return 1 - np.square(self.tanh(x))

    def grad_ReLU(self,x):
        return np.where(x > 0, 1, 0)

    def grad_output(self, x):
        return 1


class Layer():
    def __init__(self, in_units, out_units, activation, weightType, islast=False):
        np.random.seed(42)

        self.w = np.zeros((in_units + 1, out_units))
        if (weightType == 'random'):
            self.w = 0.01 * np.random.random((in_units + 1, out_units))

        self.x = None
        self.a = None
        self.z = None
        self.delta = None
        self.gradient = np.zeros((in_units + 1, out_units))
        self.velocity = np.zeros((in_units + 1, out_units))
        self.activation = activation
        self.islast = islast

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = util.append_bias(x)
        self.a = np.dot(self.x, self.w)
        self.z = self.activation.forward(self.a)

        return self.z

    def backward(self, deltaCur, learning_rate, momentum_gamma, regularization, nextLayer, gradReqd = True, isL2 = True, target = None):
        
        if self.islast:
            self.delta = target - self.z
        
        else:
            self.delta = np.dot(deltaCur, nextLayer.w[1:].T) * self.activation.backward(self.a)

        self.gradient = -np.dot(self.x.T, self.delta)

        if gradReqd:
            if isL2:
                self.velocity = momentum_gamma * self.velocity + learning_rate * (self.gradient + regularization * 2 * self.w)
                self.w = self.w - self.velocity
            
            else:
                self.velocity = momentum_gamma * self.velocity + learning_rate * (self.gradient + regularization * np.sign(self.w))
                self.w = self.w - self.velocity
        
        return self.delta

class Neuralnetwork():
    def __init__(self, config):
        self.layers = []
        self.num_layers = len(config['layer_specs']) - 1
        self.x = None
        self.y = None
        self.learning_rate = config['learning_rate']
        self.momentum_gamma = config['momentum_gamma'] if config['momentum'] else 0
        self.regularization = config['regularization']
        self.isL2 = config['isL2']

        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation(config['activation']),
                                         config["weight_type"], islast=False))
            elif i == self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation("output"),
                                         config["weight_type"], islast=True))

    def __call__(self, x, targets=None):
        return self.forward(x, targets)

    def forward(self, x):
        self.x = x
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                self.y = self.layers[i].forward(self.x)
            else:
                self.x = self.layers[i].forward(self.x)
        return self.y

    def loss(self, logits, targets = None):
        return -np.sum(targets * np.log(logits+1e-21)) / targets.shape[0]

    def backward(self, gradReqd = True, targets = None):
        delta = None
        for i in range(self.num_layers - 1, -1, -1):
            if i == self.num_layers - 1:
                delta = self.layers[i].backward(delta, self.learning_rate, self.momentum_gamma, self.regularization, None, gradReqd, self.isL2, targets)
            else:
                delta = self.layers[i].backward(delta, self.learning_rate, self.momentum_gamma, self.regularization, self.layers[i + 1], gradReqd, self.isL2, targets)
    
    def get_weight(self):
        weights = []
        for i in range(self.num_layers):
            weight = self.layers[i].w
            weights.append(weight)
        return weights

    def set_weight(self, weights):
        for i in range(self.num_layers):
            self.layers[i].w = weights[i]
        
    def epsilon_change(self, epsilon, index, add = True):
        i, j, k = index # i is the index of layer, j and k are the index of a weight value in weight matrix
        weights = self.get_weight()
        if add:
            weights[i][j][k] = weights[i][j][k] + epsilon
        else:
            weights[i][j][k] = weights[i][j][k] - epsilon
        self.set_weight(weights)