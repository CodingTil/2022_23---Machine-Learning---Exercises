import numpy as np
from numpy.random import choice

from simpleClassifier import simpleClassifier
def adaboostSimple(X, Y, K, nSamples):
    # Adaboost with decision stump classifier as weak classifier
    #
    # INPUT:
    # X         : training examples (numSamples x numDim)
    # Y         : training lables (numSamples x 1)
    # K         : number of weak classifiers to select (scalar)
    #             (the _maximal_ iteration count - possibly abort earlier
    #              when error is zero)
    # nSamples  : number of training examples which are selected in each round (scalar)
    #             The sampling needs to be weighted!
    #             Hint - look at the function 'choice' in package numpy.random
    #
    # OUTPUT:
    # alphaK 	: voting weights (K x 1) - for each round
    # para		: parameters of simple classifier (K x 2) - for each round
    #           : dimension 1 is j
    #           : dimension 2 is theta

    #####Insert your code here for subtask 1c#####
    # initialize weights (numSamples x K) - for k=0: all weights are equal, for k>0: the weights are 0
    weights = np.zeros((X.shape[0], (K+1)))
    weights[:, 0] = 1 / X.shape[0]
    # initialize alphaK (K x 1)
    alphaK = np.zeros((K, 1))
    # initialize para (K x 3)
    para = np.zeros((K, 3))

    for k in range(K):
        # sample nSamples examples from X,Y with replacement
        # the probability of each sample is given by the weights
        sampled_indices = choice(range(X.shape[0]), nSamples, p=weights[:, k])
        sampled_X = X[sampled_indices, :]
        sampled_Y = Y[sampled_indices, :]

        # train the weak classifier on the weighted data (simpleClassifier)
        j, theta, p = simpleClassifier(sampled_X, sampled_Y)
        para[k, 0] = j
        para[k, 1] = theta
        para[k, 2] = p

        # compute the error of the weak classifier
        # error = sum over all samples of the weight of the sample * I(Y_i != I(p*X_i,j < p * theta))
        error = 0
        for i in range(X.shape[0]):
            tmp = -1
            if p * X[i, j] > p * theta:
                tmp = 1
            if tmp != Y[i]:
                error += weights[i, k]

        # compute the voting weight of the weak classifier
        # alpha_k = 1/2 * ln((1-error)/error)
        alphaK[k] = 1 / 2 * np.log((1 - error) / error)

        # update the weights
        # w_i,k+1 = w_i,k * exp(-alpha_k * Y_i * I(Y_i != I(p*X_i,j < p * theta)))
        for i in range(X.shape[0]):
            tmp = -1
            if p * X[i, j] > p * theta:
                tmp = 1
            weights[i, k + 1] = weights[i, k] * np.exp(-alphaK[k] * Y[i] * tmp)

        # normalize the weights
        weights[:, k + 1] = weights[:, k + 1] / np.sum(weights[:, k + 1])

        # check if error is zero
        if error == 0:
            break

    return alphaK, para
