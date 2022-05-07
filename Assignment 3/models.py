from layer import Layer
import numpy as np

# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights = self.weights - learning_rate * weights_error
        self.bias = self.bias - learning_rate * output_error
        return input_error

# inherit from base class Layer
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

class ConvLayer(Layer):
    def __init__(self, input_shape, filter_shape, stride, padding, activation, activation_prime):
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.activation_prime = activation_prime
        self.weights = np.random.rand(filter_shape[0], filter_shape[1], input_shape[2], filter_shape[2]) - 0.5
        self.bias = np.random.rand(1, filter_shape[2]) - 0.5

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.convolve(self.input, self.weights, self.bias))
        return self.output
    
    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = self.convolve_backward(self.input, self.weights, output_error, self.bias, self.activation_prime)
        weights_error = self.convolve_backward_weights(self.input, self.weights, output_error, self.bias, self.activation_prime)
        bias_error = output_error.sum(axis=0)
        # dBias = output_error

        # update parameters
        self.weights = self.weights - learning_rate * weights_error
        self.bias = self.bias - learning_rate * bias_error
        return input_error

    # performs a convolution on the input_data with the filter_weights and bias
    def convolve(self, input_data, filter_weights, bias):
        output = np.zeros((input_data.shape[0], filter_weights.shape[2], input_data.shape[1] - filter_weights.shape[1] + 1, input_data.shape[2] - filter_weights.shape[3] + 1))
        for i in range(output.shape[0]):
            for j in range(output.shape[2]):
                for k in range(output.shape[3]):
                    output[i, j, k] = (input_data[i, j*self.stride:j*self.stride+filter_weights.shape[1], k*self.stride:k*self.stride+filter_weights.shape[3]] * filter_weights).sum() + bias[i, j, k]
        return output

    # performs a convolution on the input_data with the filter_weights and bias
    def convolve_backward(self, input_data, filter_weights, output_error, bias, activation_prime):
        input_error = np.zeros(input_data.shape)
        for i in range(input_error.shape[0]):
            for j in range(input_error.shape[1]):
                for k in range(input_error.shape[2]):
                    input_error[i, j, k] = (output_error[i, j, k] * activation_prime(input_data[i, j, k]) * filter_weights[:, :, j, k]).sum()
        return input_error

    # performs a convolution on the input_data with the filter_weights and bias
    def convolve_backward_weights(self, input_data, filter_weights, output_error, bias, activation_prime):
        weights_error = np.zeros(filter_weights.shape)
        for i in range(weights_error.shape[0]):
            for j in range(weights_error.shape[1]):
                for k in range(weights_error.shape[2]):
                    weights_error[i, j, k] = (output_error[i, j, k] * activation_prime(input_data[i, j, k]) * input_data[i, j, k]).sum()
        return weights_error

    # performs a convolution on the input_data with the filter_weights and bias
    def convolve_backward_bias(self, input_data, filter_weights, output_error, bias, activation_prime):
        bias_error = output_error.sum(axis=0)
        return bias_error

'''Instantiate
two convolutional layers having 32 and 64 3x3 filters using the self-coded convolutional layer
class. Initialize convolutional filters using Kaiming initialization.'''

class PoolingLayer(Layer):
    def __init__(self, input_shape, filter_shape, stride, padding, activation, activation_prime):
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.activation_prime = activation_prime
        self.weights = np.random.rand(filter_shape[0], filter_shape[1], input_shape[2], filter_shape[2]) - 0.5
        self.bias = np.random.rand(1, filter_shape[2]) - 0.5
    
    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.pool(self.input, self.weights, self.bias))
        return self.output
    
    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = self.pool_backward(self.input, self.weights, output_error, self.bias, self.activation_prime)
        weights_error = self.pool_backward_weights(self.input, self.weights, output_error, self.bias, self.activation_prime)
        bias_error = output_error.sum(axis=0)
        # dBias = output_error

        # update parameters
        self.weights = self.weights - learning_rate * weights_error
        self.bias = self.bias - learning_rate * bias_error
        return input_error
    
    # performs a pooling on the input_data with the filter_weights and bias
    def pool(self, input_data, filter_weights, bias):
        output = np.zeros((input_data.shape[0], filter_weights.shape[2], input_data.shape[1] - filter_weights.shape[1] + 1, input_data.shape[2] - filter_weights.shape[3] + 1))
        for i in range(output.shape[0]):
            for j in range(output.shape[2]):
                for k in range(output.shape[3]):
                    output[i, j, k] = (input_data[i, j*self.stride:j*self.stride+filter_weights.shape[1], k*self.stride:k*self.stride+filter_weights.shape[3]] * filter_weights).sum() + bias[i, j, k]
        return output
    
    # performs a pooling on the input_data with the filter_weights and bias
    def pool_backward(self, input_data, filter_weights, output_error, bias, activation_prime):
        input_error = np.zeros(input_data.shape)
        for i in range(input_error.shape[0]):
            for j in range(input_error.shape[1]):
                for k in range(input_error.shape[2]):
                    input_error[i, j, k] = (output_error[i, j, k] * activation_prime(input_data[i, j, k]) * filter_weights[:, :, j, k]).sum()
        return input_error
    
    # performs a pooling on the input_data with the filter_weights and bias
    def pool_backward_weights(self, input_data, filter_weights, output_error, bias, activation_prime):
        weights_error = np.zeros(filter_weights.shape)
        for i in range(weights_error.shape[0]):
            for j in range(weights_error.shape[1]):
                for k in range(weights_error.shape[2]):
                    weights_error[i, j, k] = (output_error[i, j, k] * activation_prime(input_data[i, j, k]) * input_data[i, j, k]).sum()
        return weights_error
    
    # performs a pooling on the input_data with the filter_weights and bias
    def pool_backward_bias(self, input_data, filter_weights, output_error, bias, activation_prime):
        bias_error = output_error.sum(axis=0)
        return bias_error
    
    # returns the activated input
    def get_output(self):
        return self.output

class FlattenLayer(Layer):
    def __init__(self):
        self.input = None
        self.output = None
        self.activation = None
        self.activation_prime = None
    
    # performs a forward propagation on the input_data
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = input_data.reshape(input_data.shape[0], -1)
        return self.output
    
    # performs a backward propagation on the input_data
    def backward_propagation(self, input_error, learning_rate):
        self.input_error = input_error
        self.output_error = input_error.reshape(self.input_error.shape[0], -1)
        return self.output_error


def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):
    return np.max(0,x)

def relu_prime(x):
    if x>0:
        return 1
    else:
        0

# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.error = []

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            self.error.append(err)
            # print('epoch %d/%d   error=%f' % (i+1, epochs, err), end = '\r')



