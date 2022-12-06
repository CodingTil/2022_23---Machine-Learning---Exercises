import numpy as np
from numpy.random import choice
from leastSquares import leastSquares

def adaboostLSLC(X, Y, K, nSamples):
    # Adaboost with least squares linear classifier as weak classifier
    # for a D-dim dataset
    #
    # INPUT:
    # X         : the dataset (numSamples x numDim)
    # Y         : labeling    (numSamples x 1)
    # K         : number of weak classifiers (iteration number of Adaboost) (scalar)
    # nSamples  : number of data which are weighted sampled (scalar)
    #
    # OUTPUT:
    # alphaK    : voting weights (K x 1)
    # para      : parameters of least square classifier (K x 3)
    #             For a D-dim dataset each least square classifier has D+1 parameters
    #             w0, w1, w2........wD

    #####Insert your code here for subtask 1e#####
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
        w, b = leastSquares(sampled_X, sampled_Y)
        para[k, 0] = w[0]
        para[k, 1] = w[1]
        para[k, 2] = b

        # compute the error of the weak classifier
        # error = sum over all samples of the weight of the sample * I(Y_i != I(p*X_i,j < p * theta))
        error = 0
        for i in range(X.shape[0]):
            tmp = -1
            if np.dot(w.T, X[i]) + b > 0:
                tmp = 1
            if tmp != Y[i]:
                error += weights[i, k]

        # compute the voting weight of the weak classifier
        # alpha_k = 1/2 * ln((1-error)/error)
        alphaK[k] = 1 / 2 * np.log((1 - error) / error)

        # update the weights
        for i in range(X.shape[0]):
            tmp = -1
            if np.dot(w.T, X[i]) + b > 0:
                tmp = 1
            if tmp != Y[i]:
                weights[i, k+1] = weights[i, k] * np.exp(alphaK[k])
            else:
                weights[i, k+1] = weights[i, k] * np.exp(-alphaK[k])

        # normalize the weights
        weights[:, k+1] = weights[:, k+1] / np.sum(weights[:, k+1])

    return [alphaK, para]
