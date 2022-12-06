import numpy as np
from numpy.random import choice
from leastSquares import leastSquares
from eval_adaBoost_leastSquare import eval_adaBoost_leastSquare


def adaboostUSPS(X, Y, K, nSamples, percent):
    # Adaboost with least squares linear classifier as weak classifier on USPS data
    # for a high dimensional dataset
    #
    # INPUT:
    # X         : the dataset (numSamples x numDim)
    # Y         : labeling    (numSamples x 1)
    # K         : number of weak classifiers (scalar)
    # nSamples  : number of data points obtained by weighted sampling (scalar)
    #
    # OUTPUT:
    # alphaK    : voting weights (1 x k)
    # para      : parameters of simple classifier (K x (D+1))
    #             For a D-dim dataset each simple classifier has D+1 parameters
    # error     : training error (1 x k)

    #####Insert your code here for subtask 1f#####
    # Sample random a percentage of data as test data set
    # The dataset consists of a matrix X and a label vector Y. Each row of the matrix X is an image of size 20 × 14 and can be viewed with matplotlib.pyplot.imshow(X[0,:].reshape(20,14,order=’F’).copy()) . The first 5000 rows of X contain the images of the digit 2, and the rest contains the images of the digit 9. Perform a random split of the 10000 data points into two equally sized subsets, one for training and one for validation. Run this at least three times and plot the cross validation error estimates vs. the number k of iterations.

    test_indices = choice(range(X.shape[0]), int(X.shape[0] * percent), replace=False)
    testX = X[test_indices, :]
    testY = Y[test_indices, :]
    trainX = np.delete(X, test_indices, axis=0)
    trainY = np.delete(Y, test_indices, axis=0)

    # initialize weights (numSamples x K) - for k=0: all weights are equal, for k>0: the weights are 0
    weights = np.zeros((trainX.shape[0], (K+1)))
    weights[:, 0] = 1 / trainX.shape[0]
    # initialize alphaK (K x 1)
    alphaK = np.zeros((K, 1))
    # initialize para (K x 3)
    para = np.zeros((K, trainX.shape[1] + 1))
    # initialize error (K x 1)
    error = np.zeros((K, 1))

    for k in range(K):
        # sample nSamples examples from X,Y with replacement
        # the probability of each sample is given by the weights
        sampled_indices = choice(range(trainX.shape[0]), nSamples, p=weights[:, k])
        sampled_X = trainX[sampled_indices, :]
        sampled_Y = trainY[sampled_indices, :]

        # train the weak classifier on the weighted data (simpleClassifier)
        w, b = leastSquares(sampled_X, sampled_Y)
        para[k, 0] = w[0]
        para[k, 1] = w[1]
        para[k, 2] = b

        # compute the error of the weak classifier
        # error = sum over all samples of the weight of the sample * I(Y_i != I(p*X_i,j < p * theta))
        e = 0
        for i in range(trainX.shape[0]):
            tmp = -1
            if np.dot(w.T, trainX[i]) + b > 0:
                tmp = 1
            if tmp != trainY[i]:
                e += weights[i, k]

        # compute the voting weight of the weak classifier
        # alpha_k = 1/2 * ln((1-error)/error)
        alphaK[k] = 1 / 2 * np.log((1 - e) / e)

        # update the weights
        for i in range(trainX.shape[0]):
            tmp = -1
            if np.dot(w.T, trainX[i]) + b > 0:
                tmp = 1
            if tmp != trainY[i]:
                weights[i, k+1] = weights[i, k] * np.exp(alphaK[k])
            else:
                weights[i, k+1] = weights[i, k] * np.exp(-alphaK[k])

        # normalize the weights
        weights[:, k+1] = weights[:, k+1] / np.sum(weights[:, k+1])

        # compute the error on the validation set
        # Hint: use the function eval_adaBoost_leastSquare
        classLabels, _ = eval_adaBoost_leastSquare(testX, alphaK[0:k+1], para[0:k+1, :])
        error[k] = np.sum(classLabels != testY) / testY.shape[0]

    return [alphaK, para, error]
