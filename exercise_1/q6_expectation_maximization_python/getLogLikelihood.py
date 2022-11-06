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
    # D=2, N=275, K=3
    D = X.shape[1]
    N = X.shape[0]
    K = len(weights)

    # sum over N data points (rows of X)
    logLikelihood = 0
    for n, x in enumerate(X):
        # sum over K Gaussians (entries in weights)
        res = 0
        for k, w in enumerate(weights):
            # compute gaussian
            gaussian = 1 / np.sqrt((2 * np.pi) ** D * np.linalg.det(covariances[:, :, k])) \
                       * np.exp(-0.5 * np.transpose(np.subtract(x, means[k]))
                                .dot(np.linalg.inv(covariances[:, :, k]))
                                .dot(np.subtract(x, means[k]))
                                )

            res += w * gaussian
        logLikelihood += np.log(res)
    return logLikelihood

