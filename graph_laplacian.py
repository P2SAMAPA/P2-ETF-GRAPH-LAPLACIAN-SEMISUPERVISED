import numpy as np

def label_propagation(returns, labels, alpha=0.2, n_neighbors=5, max_iter=100):
    """
    Iterative label propagation: f = (1-alpha)*y + alpha * (D^{-1} A f)
    """
    n = returns.shape[1]
    if n < 3:
        # Not enough assets, return uniform probability
        return np.full(n, 0.5)
    # Build graph from correlation distance
    corr = returns.corr().values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    # Build k-NN graph (symmetric)
    adj = np.zeros((n, n))
    k = min(n_neighbors, n-1)
    for i in range(n):
        nearest = np.argsort(dist[i])[1:k+1]
        adj[i, nearest] = 1
    adj = np.maximum(adj, adj.T)
    # Ensure no isolated nodes
    for i in range(n):
        if np.sum(adj[i]) == 0:
            adj[i, (i+1)%n] = 1
            adj[(i+1)%n, i] = 1
    # Degree matrix
    D = np.sum(adj, axis=1, keepdims=True)
    D_inv = 1.0 / (D + 1e-12)
    P = adj * D_inv
    # Initial scores y
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
    # Convert to probability
    prob = (f + 1) / 2
    prob = np.clip(prob, 0, 1)
    # Avoid exactly zero (to allow normalisation)
    prob = np.where(prob < 1e-6, 1e-6, prob)
    return prob

def graph_laplacian_score(returns, labels, alpha=0.2, n_neighbors=5):
    return label_propagation(returns, labels, alpha, n_neighbors)
