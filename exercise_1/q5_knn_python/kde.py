import numpy as np


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    #####Insert your code here for subtask 5a#####
    # Compute the number of samples created
    D = 1
    N = samples.size
    pos = np.arange(-5, 5.0, 0.1)

    # Init empty return array
    estDensity = np.zeros((len(pos), 2))
    # Init first column
    estDensity[:, 0] = pos

    # Iterate over first column
    for id, x in enumerate(estDensity):
        # calc probability density estimate sum for x
        sum = 0
        for i in samples:
            sum += 1 / (np.power(2 * np.pi, D/2) * h) * np.exp(-np.power(np.linalg.norm(x[0] - i), 2) / (2 * h ** 2))
        # probability density estimate = 1/N * sum
        pde = 1/N * sum

        estDensity[id, 1] = pde

    return estDensity
