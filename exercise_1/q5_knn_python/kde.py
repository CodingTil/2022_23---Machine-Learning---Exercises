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
    pos = np.arange(-5, 5, 0.1)
    estDensity = np.zeros((len(pos), 2))
    for i in range(len(pos)):
        estDensity[i, 0] = pos[i]

    for i in range(len(pos)):
        x = pos[i]
        est = 0
        for x_n in samples:
            est += k(np.abs(x - x_n), h)
        estDensity[i, 1] = est / len(samples)

    return estDensity


def k(u, h):
    return 1 / np.sqrt(2 * np.pi * (h**2)) * np.exp(-(u**2 / (2 * (h**2))))
