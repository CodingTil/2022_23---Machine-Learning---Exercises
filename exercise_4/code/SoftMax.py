"""
SoftMax Module for the Neural Network

Methods:
fprop: Forward propagation
bprop: Backward propagation
update: Update the parameters of the module
"""

import numpy as np


class SoftMax(object):
    """
    SoftMax Module for the Neural Network

    Methods:
    fprop: Forward propagation
    bprop: Backward propagation
    """

    def __init__(self):
        """
        Initialize the SoftMax Module
        """
        self.input = None
        self.output = None

    def fprop(self, input):
        """
        Forward propagation

        Args:
        input: Input to the module

        Returns:
        output: Output of the module
        """
        self.input = input
        self.output = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output /= np.sum(self.output, axis=1, keepdims=True)
        return self.output

    def bprop(self, output_grad):
        """
        Backward propagation

        Args:
        output_grad: Gradient of the output

        Returns:
        input_grad: Gradient of the input
        """
        input_grad = self.output * (
            output_grad - np.sum(output_grad * self.output, axis=1, keepdims=True)
        )
        return input_grad

    def update(self, learning_rate):
        """
        Update the parameters of the module

        Args:
        learning_rate: Learning rate of the update
        """
        pass
