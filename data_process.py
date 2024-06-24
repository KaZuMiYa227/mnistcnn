import sys, os
import cupy as cp
import numpy as np

class Functions():
    # activation function

    #sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + cp.exp(-x))
    
    # sigmoid grad function
    def sigmoid_grad(self, x):
        return (1.0 - self.sigmoid(x)) * self.sigmoid(x)
    
    # softmax function
    def softmax(self, x):
        x = x - cp.max(x, axis = -1, keepdims = True) 
        return cp.exp(x) / cp.sum(cp.exp(x), axis = -1, keepdims = True)

    # loss function

    # mean squared error
    def mean_squared_erro(seld, y, t):
        return 0.5 * cp.sum((y - t) ** 2)
    
    # cross entropy error
    def cross_entropy_error(self, y, t):
        delta = 1e-7
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, t.size)

            batch_size = y.shape[0]
            return -cp.sum(t * cp.log(y + delta)) / batch_size
        
    # nemrical differentiation
    def numerical_gradient(self, f, x):
        h = 1e-4
        grad = cp.zeros_likes(0)

        for idx in range(x.size):
            tmp_val = x[idx]

            # f(x+h)
            x[idx] = tmp_val + h
            fxh1 = f(x)

            # f(x-h)
            x[idx] = tmp_val - h
            fxh2 = f(x)
            grad[idx] = (fxh1 - fxh2) / (2*h)
            x[idx] = tmp_val
            return grad

class TwoLayerNet:
    # initialization
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * cp.random.randn(input_size, hidden_size)
        self.params["b1"] = cp.zeros(hidden_size)
        self.params["W2"] = weight_init_std * cp.random.randn(hidden_size, output_size)
        self.params["b2"] = cp.zeros(output_size)
        self.functions = Functions()

    # inference processing 
    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        # CuPy array >> NumPy array
        x = cp.array(x)

        a1 = cp.dot(x, W1) + b1
        z1 = self.functions.sigmoid(a1)
        a2 = cp.dot(z1, W2) + b2
        y = self.functions.softmax(a2)

        return y
    
    # loss function 
    # x: input data t: training data

    def loss(self, x, t):
        y = self.predict(x)

        return self.functions.cross_entropy_error(y, cp.array(t))
    
    # accuracy calculation
    def accuracy(self, x, t):
        y = self.predict(x)
        y = cp.argmax(y, axis = 1)

        # CuPy array >> NumPy array
        t = cp.array(t)
        t = cp.argmax(t, axis = 1)

        accuracy = cp.sum(y == t) / float(x.shape[0])

        return accuracy

    # numerical gradient
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads["W1"] = self.functions.numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = self.functions.numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = self.functions.numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = self.functions.numerical_gradient(loss_W, self.params["b2"])
    
        return grads
    
    # back propagation
    def gradient(self, x, t):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]
        grads = {}

        batch_num = x.shape[0]
        
        # CuPy array >> NumPy array
        x = cp.array(x)  
        t = cp.array(t)

        # forward
        a1 = cp.dot(x, W1) + b1
        z1 = self.functions.sigmoid(a1)
        a2 = cp.dot(z1, W2) + b2
        y = self.functions.softmax(a2)
        
        # CuPy array >> NumPy array
        y = cp.array(y)

        # backward
        dy = (y - t) / batch_num
        grads["W2"] = cp.dot(z1.T, dy)
        grads["b2"] = cp.sum(dy, axis = 0)
        
        dz1 = cp.dot(dy, W2.T)
        da1 = self.functions.sigmoid(a1) * dz1
        grads["W1"] = cp.dot(x.T, da1)
        grads["b1"] =  cp.sum(da1, axis = 0)

        return grads



    


