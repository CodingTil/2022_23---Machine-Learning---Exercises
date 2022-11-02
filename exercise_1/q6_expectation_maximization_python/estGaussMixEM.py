import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from EStep import EStep
from MStep import MStep
from regularize_cov import regularize_cov


def estGaussMixEM(data, K, n_iters, epsilon):
    # EM algorithm for estimation gaussian mixture mode
    #
    # INPUT:
    # data           : input data, N observations, D dimensional
    # K              : number of mixture components (modes)
    #
    # OUTPUT:
    # weights        : mixture weights - P(j) from lecture
    # means          : means of gaussians
    # covariances    : covariancesariance matrices of gaussians
    # logLikelihood  : log-likelihood of the data given the model

    #####Insert your code here for subtask 6e#####
    weights = np.ones(K) / K
    kmeans = KMeans(n_clusters=K, n_init=10).fit(data)
    cluster_idx = kmeans.labels_
    means = kmeans.cluster_centers_

    # Create initial covariance matrices
    covariances = np.zeros((data.shape[1], data.shape[1], K))
    for j in range(K):
        data_cluster = data[cluster_idx == j]
        min_dist = np.inf
        for i in range(K):
            # compute sum of distances in cluster
            dist = np.mean(euclidean_distances(data_cluster, [means[i]], squared=True))
            if dist < min_dist:
                min_dist = dist
                covariances[:, :, j] = np.eye(data.shape[1]) * min_dist

    # Run EM algorithm
    for _ in range(n_iters):
        log_likelihood, gamma = EStep(means, covariances, weights, data)
        weights, means, covariances, log_likelihood = MStep(gamma, data)
        for k in range(K):
            covariances[:, :, k] = regularize_cov(covariances[:, :, k], epsilon)

    return [weights, means, covariances]
