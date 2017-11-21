#!/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import sklearn

def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.

    Parameters
    ------------
    X: {Numpy ndarray}, shape = [n_samples, n_features]

    gamma: float
      Tuning parameter of the RBF kernel

    n_components: int
      Number of principal components to return

    Returns
    ------------
    X_pc: {Numpy ndarray}, shape = [n_samples, k_features]
      Projected dataset

    lambda: list
      Eigenvalues

    """
    # calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # numpy.eigh returns them in stored order
    eigvals, eigvecs = eigh(K)

    # Collect the top k eigenvectors (projected samples)
    alphas = np.column_stack((
        eigvecs[:, -i] for i in range(1, n_components + 1)
    ))
    # collec the corresponding eigenvalues
    lambdas = [eigvals[-i] for i in range(1, n_components+1)]

    return alphas, lambdas

def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)

def main():
    test_moons()
    test_circles()

def test_moons():
    from sklearn.datasets import make_moons
    import matplotlib.pyplot as plt
    X, y = make_moons(n_samples=100, random_state=123)
    plt.scatter(X[y==0, 0], X[y==0, 1], color='r', marker='^', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1], color='b', marker='o', alpha=0.5)
    plt.show()

    fig, ax = test_pca(X, y)
    plt.show()

    fig, ax = test_kpca(X, y)
    plt.show()

    fig, ax = test_projection(X, y, gamma=15)
    plt.show()

    fig, ax = test_skl_kpca(X, y)
    plt.show()

def test_circles():
    from sklearn.datasets import make_circles
    X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
    plt.scatter(X[y==0, 0], X[y==0, 1], color='r', marker='^', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1], color='b', marker='o', alpha=0.5)
    plt.show()

    fig, ax = test_pca(X, y)
    plt.show()
    fig, ax = test_kpca(X, y)
    plt.show()

def plot_test(X, y):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
    ax[0].scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
    ax[0].scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
    ax[1].scatter(X[y==0, 0], np.zeros((50,1))+0.02, color='red', marker='^', alpha=0.5)
    ax[1].scatter(X[y==1, 0], np.zeros((50,1))-0.02, color='blue', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')
    return fig, ax

def test_pca(X, y):
    from sklearn.decomposition import PCA
    scikit_pca = PCA(n_components=2)
    X_spca = scikit_pca.fit_transform(X)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
    print(X_spca.shape)
    print(X_spca[y==0].shape)
    print(X_spca[y==1].shape)
    n = len(X_spca[y==0, 0])
    ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1], color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1], color='blue', marker='o', alpha=0.5)
    ax[1].scatter(X_spca[y==0, 0], np.zeros((n,1))+0.02, color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_spca[y==1, 0], np.zeros((n,1))-0.02, color='blue', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')

    return fig, ax

def test_kpca(X, y, gamma=15):
    from matplotlib.ticker import FormatStrFormatter
    alphas, lambdas = rbf_kernel_pca(X, gamma=gamma, n_components=2)
    X_kpca = alphas
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
    print(X_kpca.shape)
    print(X_kpca[y==0].shape)
    print(X_kpca[y==1].shape)
    n = len(X_kpca[y==0, 0])
    ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='r', marker='^', alpha=0.5)
    ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='b', marker='o', alpha=0.5)
    ax[1].scatter(X_kpca[y==0, 0], np.zeros([n,1])+0.02, color='r', marker='^', alpha=0.5)
    ax[1].scatter(X_kpca[y==1, 0], np.zeros([n,1])-0.02, color='b', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')
    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

    return fig, ax

def test_projection(X, y, gamma=15):
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=100, random_state=123)
    alphas, lambdas = rbf_kernel_pca(X, gamma=gamma, n_components=1)
    x_new = X[25]
    x_proj = alphas[25] # original projection
    x_reproj = project_x(x_new, X, gamma=gamma, alphas=alphas, lambdas=lambdas)
    print(x_proj)
    print(x_reproj)
    plt.scatter(alphas[y==0, 0], np.zeros((50)), color='r', marker='^', alpha=0.5)
    plt.scatter(alphas[y==1, 0], np.zeros((50)), color='b', marker='o', alpha=0.5)
    plt.scatter(x_proj, 0, color='k', label='original projection of point X[25]', marker='^', s=100)
    plt.scatter(x_reproj, 0, color='g', label='remapped point X[25]', marker='x', s=500)
    plt.legend(scatterpoints=1)

    return plt.gcf(), plt.gca()

def test_skl_kpca(X, y):
    from sklearn.decomposition import KernelPCA
    scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
    X_skernpca = scikit_kpca.fit_transform(X)
    plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], color='r', marker='^', alpha=0.5)
    plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], color='b', marker='o', alpha=0.5)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    return plt.gcf(), plt.gca()

if __name__=='__main__':
    main()
