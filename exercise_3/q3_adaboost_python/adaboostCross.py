import numpy as np
from numpy.random import choice
from simpleClassifier import simpleClassifier
from eval_adaBoost_simpleClassifier import eval_adaBoost_simpleClassifier

def adaboostCross(X, Y, K, nSamples, percent):
    # Adaboost with an additional cross validation routine
    #
    # INPUT:
    # X         : training examples (numSamples x numDims )
    # Y         : training lables (numSamples x 1)
    # K         : number of weak classifiers to select (scalar)
    #             (the _maximal_ iteration count - possibly abort earlier)
    # nSamples  : number of training examples which are selected in each round. (scalar)
    #             The sampling needs to be weighted!
    # percent   : percentage of the data set that is used as test data set (scalar)
    #
    # OUTPUT:
    # alphaK    : voting weights (K x 1)
    # para      : parameters of simple classifier (K x 2)
    # testX     : test dataset (numTestSamples x numDim)
    # testY     : test labels  (numTestSamples x 1)
    # error	    : error rate on validation set after each of the K iterations (K x 1)

    #####Insert your code here for subtask 1d#####
    # Randomly sample a percentage of the data as test data set
    # The remaining data is used for training
    # Hint: look at the function 'choice' in package numpy.random
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
    para = np.zeros((K, 3))
    # initialize error (K x 1)
    error = np.zeros((K, 1))

    for k in range(K):
        # sample nSamples examples from X,Y with replacement
        # the probability of each sample is given by the weights
        sampled_indices = choice(range(trainX.shape[0]), nSamples, p=weights[:, k])
        sampled_X = trainX[sampled_indices, :]
        sampled_Y = trainY[sampled_indices, :]

        # train the weak classifier on the weighted data (simpleClassifier)
        j, theta, p = simpleClassifier(sampled_X, sampled_Y)
        para[k, 0] = j
        para[k, 1] = theta
        para[k, 2] = p

        # compute the error of the weak classifier
        # error = sum over all samples of the weight of the sample * I(Y_i != I(p*X_i,j < p * theta))
        e = 0
        for i in range(trainX.shape[0]):
            tmp = -1
            if p * trainX[i, j] > p * theta:
                tmp = 1
            if tmp != trainY[i]:
                e += weights[i, k]

        # compute the voting weight of the weak classifier
        # alpha_k = 1/2 * ln((1-error)/error)
        alphaK[k] = 1 / 2 * np.log((1 - e) / e)

        # update the weights
        for i in range(trainX.shape[0]):
            tmp = -1
            if p * trainX[i, j] > p * theta:
                tmp = 1
            if tmp == trainY[i]:
                weights[i, k+1] = weights[i, k] * np.exp(-alphaK[k])
            else:
                weights[i, k+1] = weights[i, k] * np.exp(alphaK[k])

        # normalize the weights
        weights[:, k+1] = weights[:, k+1] / np.sum(weights[:, k+1])

        # compute the error on the validation set
        # Hint: use the function eval_adaBoost_simpleClassifier
        classLabels, _ = eval_adaBoost_simpleClassifier(testX, alphaK[0:k+1], para[0:k+1, :])
        error[k] = np.sum(classLabels != testY) / testY.shape[0]

    return alphaK, para, testX, testY, error

