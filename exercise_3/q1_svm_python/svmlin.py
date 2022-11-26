import numpy as np
# might need to add path to mingw-w64/bin for cvxopt to work
# import os
# os.environ["PATH"] += os.pathsep + ...
import cvxopt


def svmlin(X, t, C):
    # Linear SVM Classifier
    #
    # INPUT:
    # X        : the dataset                  (num_samples x dim)
    # t        : labeling                     (num_samples x 1)
    # C        : penalty factor for slack variables (scalar)
    #
    # OUTPUT:
    # alpha    : output of quadprog function  (num_samples x 1)
    # sv       : support vectors (boolean)    (1 x num_samples)
    # w        : parameters of the classifier (1 x dim)
    # b        : bias of the classifier       (scalar)
    # result   : result of classification     (1 x num_samples)
    # slack    : points inside the margin (boolean)   (1 x num_samples)

    #####Insert your code here for subtask 2a#####
    N = X.shape[0]
    q = (-1) * np.ones(N)
    # H(n, m) = tntm〈xn, xm〉
    H = np.zeros((N, N))
    for n in range(N):
        for m in range(N):
            H[n, m] = t[n] * t[m] * X[n].dot(X[m])
    n = H.shape[1]
    G = np.vstack([-np.eye(n), np.eye(n)])
    A = t
    b = np.zeros(1)
    h = np.hstack([np.zeros(N), C * np.ones(N)])

    P = cvxopt.matrix(H)
    q = cvxopt.matrix(q)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    A = cvxopt.matrix(A, (1, N))
    b = cvxopt.matrix(b)
    a = cvxopt.solvers.qp(P, q, G, h, A, b)

    alpha = np.array(a['x'])
    sv = (alpha > 1e-5).T
    w = np.sum(np.mat(alpha) * np.mat(t.T) * X, axis=0)
    b = np.mean(t[sv.flatten()] - X[sv.flatten()].dot(w.T))
    result = np.sign(X.dot(w.T) + b).T
    slack = np.logical_and(result.flatten() == t.flatten(), (alpha < C).flatten())

    alpha = np.asarray(alpha)
    sv = np.asarray(sv)
    w = np.asarray(w)
    result = np.asarray(result)
    slack = np.asarray(slack)

    assert alpha.shape == (X.shape[0], 1)
    assert sv.shape == (1, X.shape[0])
    assert w.shape == (1, X.shape[1])
    assert result.shape == (1, X.shape[0])
    assert slack.shape == (1, X.shape[0])

    return alpha, sv, w, b, result, slack
