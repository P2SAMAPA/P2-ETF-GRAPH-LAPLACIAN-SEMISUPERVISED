import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.sparse import identity, csr_matrix
from scipy.sparse.linalg import spsolve

def label_propagation(returns, labels, alpha=0.2, n_neighbors=5):
    """
    returns: DataFrame of returns (T x n) – we use correlation of the whole window.
    labels: binary array (n,) where 1 = positive, 0 = negative, -1 = unlabeled.
    """
    n = returns.shape[1]
    # Build graph from correlation distance
    corr = returns.corr().values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    # Build k-NN graph manually
    adj = np.zeros((n, n))
    for i in range(n):
        nearest = np.argsort(dist[i])[1:n_neighbors+1]
        adj[i, nearest] = 1
    adj = np.maximum(adj, adj.T)
    # Compute Laplacian
    L = laplacian(adj, normed=False)
    # Regularised Laplacian: I + alpha L
    I = identity(n, format='csr')
    A = I + alpha * L
    # Convert to CSC for spsolve
    A = A.tocsc()
    # Initial labels: +1 for positive, -1 for negative, 0 for unlabeled
    y = np.zeros(n)
    y[labels == 1] = 1.0
    y[labels == -1] = -1.0
    # Solve (I + alpha L) f = y
    try:
        f = spsolve(A, y)
    except Exception as e:
        # fallback: use pseudo-inverse or return zeros
        print(f"Warning: spsolve failed ({e}), returning zeros")
        f = np.zeros(n)
    # Convert to probability of positive class: p = (f + 1) / 2
    prob = (f + 1) / 2
    # Clip to [0,1]
    prob = np.clip(prob, 0, 1)
    return prob

def graph_laplacian_score(returns, labels, alpha=0.2, n_neighbors=5):
    return label_propagation(returns, labels, alpha, n_neighbors)
