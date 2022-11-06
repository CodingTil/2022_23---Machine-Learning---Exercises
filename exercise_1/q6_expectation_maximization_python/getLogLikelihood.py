import numpy as np


def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    #####Insert your code here for subtask 6a#####
    logLikelihood = 0
    for x in X:
        logLikelihood += np.log(mixtureLikelihood(means, weights, covariances, x))
    return logLikelihood


def mixtureLikelihood(means, weights, covariances, x):
    return sum(
        weights[k]
        * gaussianLikelihood(means[k], covariances[:, :, k], x, len(means[k]))
        for k in range(len(weights))
    )


def gaussianLikelihood(mean, covariance, x, D):
    return (
        1
        / np.sqrt((2 * np.pi) ** D * np.linalg.det(covariance))
        * np.exp(
            -0.5
            * np.transpose(np.subtract(x, mean))
            .dot(np.linalg.inv(covariance))
            .dot(np.subtract(x, mean))
        )
    )
