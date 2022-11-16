import numpy as np


def linclass(weight, bias, data):
    # Linear Classifier
    #
    # INPUT:
    # weight      : weights                (dim x 1)
    # bias        : bias term              (scalar)
    # data        : Input to be classified (num_samples x dim)
    #
    # OUTPUT:
    # class_pred       : Predicted class (+-1) values  (num_samples x 1)

    #####Insert your code here for subtask 1b#####
    # Perform linear classification i.e. class prediction

    # Extend each datapoint x as [1, x]
    data_extended = np.hstack((np.ones((data.shape[0], 1)), data))

    # Concatenate bias to weights
    weights = np.concatenate(([bias], weight))

    # Compute dot product of transpose of weights and every data entry
    class_pred = list()
    for i in range(data_extended.shape[0]):
        class_pred.append(np.sign(np.dot(np.transpose(weights), data_extended[i])))

    return class_pred
