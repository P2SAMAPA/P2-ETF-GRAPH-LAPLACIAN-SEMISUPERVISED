import numpy as np

def label_propagation(returns, labels, alpha=0.2, n_neighbors=5, max_iter=100):
    """
    Iterative label propagation: f = (1-alpha)*y + alpha * (D^{-1} A f)
    """
    n = returns.shape[1]
    # Build graph from correlation distance
    corr = returns.corr().values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    # Build k-NN graph (symmetric)
    adj = np.zeros((n, n))
    for i in range(n):
        nearest = np.argsort(dist[i])[1:n_neighbors+1]
        adj[i, nearest] = 1
    adj = np.maximum(adj, adj.T)
    # Degree matrix
    D = np.sum(adj, axis=1, keepdims=True)
    D_inv = 1.0 / (D + 1e-12)
    # Normalized adjacency: P = D^{-1} A
    P = adj * D_inv
    # Initial scores: y (positive=1, negative=-1, zero otherwise)
    y = np.zeros(n)
    y[labels == 1] = 1.0
    y[labels == -1] = -1.0
    # Iterative propagation
    f = y.copy()
    for _ in range(max_iter):
        f_new = (1 - alpha) * y + alpha * (P @ f)
        if np.allclose(f, f_new, atol=1e-8):
            break
        f = f_new
    # Convert to probability: (f + 1)/2
    prob = (f + 1) / 2
    prob = np.clip(prob, 0, 1)
    return prob

def graph_laplacian_score(returns, labels, alpha=0.2, n_neighbors=5):
    return label_propagation(returns, labels, alpha, n_neighbors)
