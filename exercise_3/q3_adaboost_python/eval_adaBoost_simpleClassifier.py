import numpy as np


def eval_adaBoost_simpleClassifier(X, alphaK, para):
    # INPUT:
    # para	: parameters of simple classifier (K x 2) - for each round
    #           : dimension 1 is j
    #           : dimension 2 is theta
    # alphaK    : classifier voting weights (K x 1)
    # X         : test data points (numSamples x numDim)
    #
    # OUTPUT:
    # classLabels: labels for data points (numSamples x 1)
    # result     : weighted sum of all the K classifier (numSamples x 1)

    #####Insert your code here for subtask 1c#####
    # initialize classLabels (numSamples x 1)
    classLabels = np.zeros((X.shape[0], 1))
    # initialize result (numSamples x 1)
    result = np.zeros((X.shape[0], 1))

    for k in range(para.shape[0]):
        # compute the result of the k-th classifier
        # result = result + alpha_k * I(p*X_i,j < p * theta)
        for i in range(X.shape[0]):
            tmp = -1
            if para[k, 2] * X[i, int(para[k, 0])] > para[k, 2] * para[k, 1]:
                tmp = 1
            result[i] += alphaK[k] * tmp

        # compute the class label for each sample
        # classLabels = sign(result)
        for i in range(X.shape[0]):
            if result[i] > 0:
                classLabels[i] = 1
            else:
                classLabels[i] = -1

    return classLabels, result
