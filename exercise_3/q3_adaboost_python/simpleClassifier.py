import numpy as np


def simpleClassifier(X, Y):
    # Select a simple classifier
    #
    # INPUT:
    # X         : training examples (numSamples x numDim) [2D real value]
    # Y         : training lables (numSamples x 1) [-1 or 1]
    #
    # OUTPUT:
    # theta 	: threshold value for the decision (scalar)
    # j 		: the dimension to "look at" (scalar)

    #####Insert your code here for subtask 1b#####
    # res : (j, theta, value)
    res_dict = dict(j=None, theta=None, value=2)

    # For both j=1 and j=2 sample n random thetas and determine the best one
    min = 2 # set impossibly high so will definitively be updated in first round
    for j in range(2):
        # select random theta and test c(X, j, theta)
        for k in range(100):
            sum = 0
            theta = rnd() # init theta
            for i in range(X.shape[0]):
                sum += I(X[i], j, theta, Y[i])
            sum /= X.shape[0]
            if sum < min:  # update min arg
                min = sum
                res_dict.update({'j': j, 'theta': theta, 'value': sum})
                print("Minimum: ", res_dict)
    j = res_dict['j']
    theta = res_dict['theta']
    return j, theta
    # instead go through min-max range stepwise

def c(x, j, theta):
    p = 1
    if p*x[j] > p*theta:
        return 1
    else:
        return -1

def I(x, j, theta, y):
    if c(x, j, theta) != y:
        return 1
    else:
        return 0

def rnd():
    if np.random.rand() <= 0.5:
        return -np.log(np.random.rand())
    else:
        return np.log(np.random.rand())
