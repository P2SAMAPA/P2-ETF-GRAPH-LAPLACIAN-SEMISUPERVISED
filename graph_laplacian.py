import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.sparse import identity, csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

def label_propagation(returns, labels, alpha=0.2, n_neighbors=5):
    n = returns.shape[1]
    corr = returns.corr().values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    # Build adjacency matrix (dense for simplicity, then convert to sparse)
    adj = np.zeros((n, n))
    for i in range(n):
        nearest = np.argsort(dist[i])[1:n_neighbors+1]
        adj[i, nearest] = 1
    adj = np.maximum(adj, adj.T)
    # Convert to sparse CSR
    adj_sparse = csr_matrix(adj)
    # Compute Laplacian (returns sparse)
    L = laplacian(adj_sparse, normed=False)
    # Identity as sparse
    I = identity(n, format='csr')
    # A = I + alpha * L
    A = I + alpha * L
    # Convert to CSC for spsolve
    A = A.tocsc()
    # Initial labels
    y = np.zeros(n)
    y[labels == 1] = 1.0
    y[labels == -1] = -1.0
    try:
        f = spsolve(A, y)
    except Exception as e:
        print(f"Warning: spsolve failed ({e}), returning zeros")
        f = np.zeros(n)
    prob = (f + 1) / 2
    prob = np.clip(prob, 0, 1)
    return prob

def graph_laplacian_score(returns, labels, alpha=0.2, n_neighbors=5):
    return label_propagation(returns, labels, alpha, n_neighbors)
