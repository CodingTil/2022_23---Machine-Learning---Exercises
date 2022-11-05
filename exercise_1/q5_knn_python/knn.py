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
    D = 1
    N = samples.size
    pos = np.arange(-5, 5.0, 0.1)

    # Init empty return array
    estDensity = np.zeros((len(pos), 2))
    # Init first column
    estDensity[:, 0] = pos

    # Find k-nearest-neighbours and store volume V* of the hyperball containing all k points
    for id, x in enumerate(estDensity):
        # Find distance k-nearest data point
        #   Create list of distances between x[0] and {samples}
        distance_list = []
        for i in samples:
            distance_list.append(abs(x[0]-i))
        #   Sort this list
        distance_list.sort()
        print(distance_list[k-1])
        #   Save V* = 2*"3rd smallest distance"
        V_star = 2*distance_list[k-1]

        estDensity[id, 1] = k / (N*V_star)

    return estDensity
