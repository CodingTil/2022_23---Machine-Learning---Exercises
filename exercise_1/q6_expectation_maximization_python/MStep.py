import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Insert your code here for subtask 6c#####
    N, K = gamma.shape
    N_hat = np.zeros(K)
    for k in range(K):
        N_hat[k] = np.sum(gamma[:, k])
    weights = N_hat / N
    means = np.zeros((K, X.shape[1]))
    for k in range(K):
        for i in range(N):
            means[k] += gamma[i, k] * X[i]
        means[k] /= N_hat[k]
    covariances = np.zeros((X.shape[1], X.shape[1], K))
    for k in range(K):
        for i in range(N):
            covariances[:, :, k] += gamma[i, k] * np.outer(X[i] - means[k], X[i] - means[k])
        covariances[:, :, k] /= N_hat[k]
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    return weights, means, covariances, logLikelihood
