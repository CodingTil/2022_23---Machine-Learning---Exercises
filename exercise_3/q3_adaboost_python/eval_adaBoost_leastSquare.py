import numpy as np

def eval_adaBoost_leastSquare(X, alphaK, para):
    # INPUT:
    # para		: parameters of simple classifier (K x (D +1))
    #           : dimension 1 is w0
    #           : dimension 2 is w1
    #           : dimension 3 is w2
    #             and so on
    # alphaK    : classifier voting weights (K x 1)
    # X         : test data points (numSamples x numDim)
    #
    # OUTPUT:
    # classLabels: labels for data points (numSamples x 1)
    # result     : weighted sum of all the K classifier (scalar)

    #####Insert your code here for subtask 1e#####
    # initialize classLabels (numSamples x 1)
    classLabels = np.zeros((X.shape[0], 1))
    # initialize result (numSamples x 1)
    result = np.zeros((X.shape[0], 1))

    for k in range(para.shape[0]):
        # compute the result of the k-th classifier
        # result = result + alpha_k * I(p*X_i,j < p * theta)
        for i in range(X.shape[0]):
            tmp = -1
            if np.dot(para[k, :], np.append(1, X[i, :])) > 0:
                tmp = 1
            result[i] += alphaK[k] * tmp

        # compute the class label for each sample
        # classLabels = sign(result)
        for i in range(X.shape[0]):
            if result[i] > 0:
                classLabels[i] = 1
            else:
                classLabels[i] = -1

    return [classLabels, result]

