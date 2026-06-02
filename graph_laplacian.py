import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import laplacian

def label_propagation(returns, labels, alpha=0.2, n_neighbors=5):
    n = returns.shape[1]
    # correlation distance
    corr = returns.corr().values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    # Build k-NN graph
    adj = np.zeros((n, n))
    # Ensure at least 2 neighbors to avoid disconnected graph
    k = min(n_neighbors, n-1)
    for i in range(n):
        nearest = np.argsort(dist[i])[1:k+1]
        adj[i, nearest] = 1
    adj = np.maximum(adj, adj.T)
    # Convert to sparse
    adj_sparse = csr_matrix(adj)
    # Compute Laplacian
    L = laplacian(adj_sparse, normed=False)
    # Regularise: I + alpha L
    I = identity(n, format='csr')
    A = I + alpha * L
    A = A.tocsc()
    y = np.zeros(n)
    y[labels == 1] = 1.0
    y[labels == -1] = -1.0
    try:
        f = spsolve(A, y)
    except Exception as e:
        # Fallback: solve dense (for small n)
        print(f"Warning: spsolve failed ({e}), falling back to dense solve")
        A_dense = A.toarray()
        try:
            f = np.linalg.solve(A_dense + 1e-8 * np.eye(n), y)
        except:
            f = np.zeros(n)
    prob = (f + 1) / 2
    prob = np.clip(prob, 0, 1)
    return prob

def graph_laplacian_score(returns, labels, alpha=0.2, n_neighbors=5):
    return label_propagation(returns, labels, alpha, n_neighbors)
