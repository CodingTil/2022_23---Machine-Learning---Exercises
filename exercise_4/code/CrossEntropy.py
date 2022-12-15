"""
CrossEntropy Module for the Neural Network

Methods:
fprop: Forward propagation
bprop: Backward propagation
update: Update the parameters of the module
"""

import numpy as np


class CrossEntropy(object):
    """
    CrossEntropy Module for the Neural Network

    Methods:
    fprop: Forward propagation
    bprop: Backward propagation
    """

    def __init__(self):
        """
        Initialize the CrossEntropy Module
        """
        self.input = None
        self.target = None
        self.output = None

    def fprop(self, input, target):
        """
        Forward propagation

        Args:
        input: Input to the module
        target: Target of the module

        Returns:
        output: Output of the module
        """
        self.input = input
        self.target = target
        self.output = -np.sum(target * np.log(input)) / input.shape[0]
        return self.output

    def bprop(self, output_grad):
        """
        Backward propagation

        Returns:
        input_grad: Gradient of the input
        """
        input_grad = -self.target / self.input
        input_grad /= self.input.shape[0]
        return input_grad

    def update(self, learning_rate):
        """
        Update the parameters of the module
        """
        pass
