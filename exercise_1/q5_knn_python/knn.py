import numpy as np


def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    #####Insert your code here for subtask 5b#####
    # Compute the number of the samples created
    pos = np.arange(-5, 5, 0.1)
    estDensity = np.zeros((len(pos), 2))
    for i in range(len(pos)):
        estDensity[i, 0] = pos[i]

    for i in range(len(pos)):
        x = pos[i]
        estDensity[i, 1] = k / (len(samples) * V(x, samples, k))

    return estDensity


def V(x, samples, k):
    sorted_samples = sorted(samples, key=lambda x_n: np.abs(x - x_n))
    return np.abs(x - sorted_samples[k - 1]) * 2
