import numpy as np
from getLogLikelihood import getLogLikelihood, gaussianLikelihood


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    gamma = np.zeros((len(X), len(weights)))
    for i in range(len(X)):
        for k in range(len(weights)):
            gamma[i, k] = (
                weights[k]
                * gaussianLikelihood(means[k], covariances[:, :, k], X[i], len(means[k]))
            )
        gamma[i] /= np.sum(gamma[i])
    return [logLikelihood, gamma]
