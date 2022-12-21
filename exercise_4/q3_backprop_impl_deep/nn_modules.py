import abc
import numpy as np


class NNModule:
    """ Class defining abstract interface every module has to implement

    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fprop(self, input):
        """ Forwardpropagate the input through the module

        :param input: Input tensor for the module
        :return Output tensor after module application
        """
        return

    @abc.abstractmethod
    def bprop(self, grad_out):
        """ Backpropagate the gradient the output to the input

        :param grad_out: Gradients at the output of the module
        :return: Gradient wrt. input
        """
        return

    @abc.abstractmethod
    def get_grad_param(self, grad_out):
        """ Return gradients wrt. the parameters
        Calculate the gardients wrt. to the parameters of the module. Function already
        accumulates gradients over the batch -> Save memory and implementation issues using numpy avoid loops

        :param grad_out: Gradients at the output
        :return: Gradients wrt. the internal parameter accumulated over the batch
        """
        return

    @abc.abstractmethod
    def apply_parameter_update(self, acc_grad_para, up_fun):
        """ Apply the update function to the internal parameters.

        :param acc_grad_para: Accumulated gradients over the batch
        :param up_fun: Update function used
        :return:
        """
        return

    # If we would like to support different initialization techniques, we could
    # use an Initializer class
    # For simplicity use a fixed initialize for each module
    @abc.abstractmethod
    def initialize_parameter(self):
        """ Initialize the internal parameter

        :return:
        """


class NNModuleParaFree(NNModule):
    """Specialization of the NNModule for modules which do not have any internal parameters

    """
    __metaclass__ = abc.ABCMeta

    def initialize_parameter(self):
        # No initialization necessary
        return

    def get_grad_param(self, grad_out):
        # No parameter gradients
        return None

    def apply_parameter_update(self, acc_grad_para, up_fun):
        # No parameters to update
        return


class LossModule(NNModule):
    """Specialization of NNModule for losses which need target values

    """
    __metaclass__ = abc.ABCMeta

    def set_targets(self, t):
        """Saves expected targets.
        Does not copy the input.

        :param t: Expected target values.
        :return:
        """
        self.t = t

    def initialize_parameter(self):
        # No internal parameters
        return

    def get_grad_param(self, grad_out):
        # No gradient for internal parameter
        return None

    def apply_parameter_update(self, acc_grad_para, up_fun):
        # No update needed
        return


# Task 2 a)
class Linear(NNModule):
    """Module which implements a linear layer"""

    #####Insert your code here for subtask 2a#####
    def __init__(self, n_in, n_out):
        self.input_dim = n_in
        self.output_dim = n_out
        self.W = np.random.randn(n_in, n_out) * 0.1
        self.b = np.zeros((1, n_out))
        self.dW = np.zeros((n_in, n_out))
        self.db = np.zeros((1, n_out))

    def fprop(self, input):
        self.cache_in = input
        return np.dot(input, self.W) + self.b

    def bprop(self, grad_out):
        self.dW = np.dot(self.cache_in.T, grad_out)
        self.db = np.sum(grad_out, axis=0)
        return np.dot(grad_out, self.W.T)

    def get_grad_param(self, grad_out):
        return self.dW, self.db

    def apply_parameter_update(self, acc_grad_para, up_fun):
        self.W = up_fun(self.W, acc_grad_para[0])
        self.b = up_fun(self.b, acc_grad_para[1])

    def initialize_parameter(self):
        self.W = np.random.randn(self.input_dim, self.output_dim) * 0.1
        self.b = np.zeros((1, self.output_dim))
        self.dW = np.zeros((self.input_dim, self.output_dim))
        self.db = np.zeros((1, self.output_dim))


# Task 2 b)
class Softmax(NNModuleParaFree):
    """Softmax layer"""

    #####Insert your code here for subtask 2b#####
    def __init__(self):
        self.cache_in = None
        self.cache_out = None

    def fprop(self, input):
        self.cache_in = input
        # substract max to avoid overflow
        input -= np.max(input, axis=1, keepdims=True)
        exp = np.exp(input)
        self.cache_out = exp / np.sum(exp, axis=1, keepdims=True)
        return self.cache_out

    def bprop(self, grad_out):
        grad_in = np.zeros_like(self.cache_in)
        for i in range(self.cache_in.shape[0]):
            grad_in[i] = self.cache_out[i] * (grad_out[i] - np.sum(grad_out[i] * self.cache_out[i]))
        return grad_in


# Task 2 c)
class CrossEntropyLoss(LossModule):
    """Cross-Entropy-Loss-Module"""
    def __init__(self):
        # Save input for bprop
        self.cache_in = None

    def fprop(self, input):
        self.cache_in = np.array(input)
        sz_batch = input.shape[0]
        loss = -1 * np.log(input[np.arange(sz_batch), self.t])
        return loss

    def bprop(self, grad_out):
        sz_batch, n_in = self.cache_in.shape
        z = np.zeros((sz_batch, n_in))
        z[np.arange(sz_batch), self.t] =  \
            -1 * 1.0/self.cache_in[np.arange(sz_batch), self.t]
        np.multiply(grad_out, z, z)
        return z


# Task 3 b)
class Tanh(NNModuleParaFree):
    """Module implementing a Tanh acitivation function"""

    def __init__(self):
        # Cache output for bprop
        self.cache_out = None

    def fprop(self, input):
        output = np.tanh(input)
        self.cache_out = np.array(output)
        return output

    def bprop(self, grad_out):
        return np.multiply(grad_out, 1 - self.cache_out ** 2)


# Task 4 e)
class LogCrossEntropyLoss(LossModule):
    """Log-Cross-Entropy-Loss"""
    def __init__(self):
        self.sz_batch = self.n_in = None

    def fprop(self, input):
        self.sz_batch, self.n_in = input.shape
        loss = -1 * input[np.arange(self.sz_batch), self.t]
        return loss

    def bprop(self, grad_out):
        z = np.zeros((self.sz_batch, self.n_in))
        z[np.arange(self.sz_batch), self.t] = -1
        np.multiply(grad_out, z, z)
        return z


# Task 4 e)
class LogSoftmax(NNModuleParaFree):
    """Log-Softmax-Module"""

    def __init__(self):
        # Save output for bprop
        self.cache_out = None

    def fprop(self, input):
        # See 4a for stability reasons
        inp_max = np.max(input, 1)
        # Transpose for numpy broadcasting -> Subtract each batch max from the batch
        input = (input.T - inp_max).T
        exponentials = np.exp(input)
        log_normalization = np.log(np.sum(exponentials, 1))

        # Transpose -> Subtract log normalization for each batch and reshape to batch \times output
        output = (input.T - log_normalization).T
        self.cache_out = np.array(output)

        return output

    def bprop(self, grad_out):
        sz_batch, n_in = grad_out.shape
        sum_grad = np.sum(grad_out, 1).reshape((sz_batch, 1))
        sigma = np.exp(self.cache_out)
        z = grad_out - np.multiply(sum_grad, sigma)
        return z
