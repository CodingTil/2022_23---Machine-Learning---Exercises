import numpy as np
from estGaussMixEM import estGaussMixEM
from getLogLikelihood import getLogLikelihood, mixtureLikelihood


def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel

    #####Insert your code here for subtask 1g#####
    weights_s, means_s, covariances_s = estGaussMixEM(sdata, K, n_iter, epsilon)
    weights_n, means_n, covariances_n = estGaussMixEM(ndata, K, n_iter, epsilon)
    result = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            p_s = mixtureLikelihood(means_s, weights_s, covariances_s, img[i, j])
            p_n = mixtureLikelihood(means_n, weights_n, covariances_n, img[i, j])
            if p_s / p_n > theta:
                result[i, j] = 1
            else:
                result[i, j] = 0
    return result
