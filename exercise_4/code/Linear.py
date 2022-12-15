"""
Linear Module for the Neural Network

Methods:
fprop: Forward propagation
bprop: Backward propagation
update: Update the parameters of the module
"""

import numpy as np


class Linear(object):
    """
    Linear Module for the Neural Network

    Methods:
    fprop: Forward propagation
    bprop: Backward propagation
    """

    def __init__(self, input_dim, output_dim):
        """
        Initialize the Linear Module

        Args:
        input_dim: Input dimension
        output_dim: Output dimension
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = np.random.randn(input_dim, output_dim) * 0.1
        self.b = np.zeros((1, output_dim))
        self.dW = np.zeros((input_dim, output_dim))
        self.db = np.zeros((1, output_dim))

    def fprop(self, input):
        """
        Forward propagation

        Args:
        input: Input to the module

        Returns:
        output: Output of the module
        """
        self.input = input
        self.output = np.dot(input, self.W) + self.b
        return self.output

    def bprop(self, output_grad):
        """
        Backward propagation

        Args:
        output_grad: Gradient of the output

        Returns:
        input_grad: Gradient of the input
        """
        self.dW = np.dot(self.input.T, output_grad)
        self.db = np.sum(output_grad, axis=0, keepdims=True)
        input_grad = np.dot(output_grad, self.W.T)
        return input_grad

    def update(self, learning_rate):
        """
        Update the parameters of the module

        Args:
        learning_rate: Learning rate
        """
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
