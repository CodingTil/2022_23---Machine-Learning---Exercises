import numpy as np


def simpleClassifier(X, Y):
    # Select a simple classifier
    #
    # INPUT:
    # X         : training examples (numSamples x numDim)
    # Y         : training lables (numSamples x 1)
    #
    # OUTPUT:
    # theta 	: threshold value for the decision (scalar)
    # j 		: the dimension to "look at" (scalar)

    #####Insert your code here for subtask 1b#####

    # create a 3d matrix:
    # 1st dimension: j - size is numDim
    # 2nd dimension: theta - size is numSamples
    # 3rd dimension: p - size is - either 1 or -1
    # entries: the error of the classifier - sum over inputs I(Y_i != I(p*X_i,j < p * theta))
    matrix = np.zeros((X.shape[1], X.shape[0], 2))
    # loop over all dimensions
    for j in range(X.shape[1]):
        for theta_index in range(X.shape[0]):
            theta = X[theta_index, j]
            for p_index in range(len([-1, 1])):
                p = [-1, 1][p_index]
                value = 0
                for i in range(X.shape[0]):
                    # c function
                    tmp = -1
                    if p*X[i,j] > p*theta:
                        tmp = 1
                    if tmp != Y[i]:
                        value += 1
                matrix[j, theta_index, p_index] = value

    # find the indices (j,theta_index, p_index) of the minimum error (argmin)
    min_index = np.unravel_index(np.argmin(matrix, axis=None), matrix.shape)
    j = min_index[0]
    theta_index = min_index[1]
    p_index = min_index[2]
    theta = X[theta_index, j]
    p = [-1, 1][p_index]
    return j, theta, p
