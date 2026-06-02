import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.sparse import identity, csr_matrix
from scipy.sparse.linalg import spsolve

def label_propagation(returns, labels, alpha=0.2, n_neighbors=5):
    """
    Propagate labels through the k‑NN graph of correlation distances.
    Returns probability of positive class for each ETF.
    """
    corr = returns.corr().values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    n = dist.shape[0]
    # Build k‑NN graph (symmetric)
    adj = np.zeros((n, n))
    for i in range(n):
        nearest = np.argsort(dist[i])[1:n_neighbors+1]
        adj[i, nearest] = 1
    adj = np.maximum(adj, adj.T)
    # Compute graph Laplacian (sparse)
    L = laplacian(adj, normed=False)
    # Regularised system: (I + α L) f = y
    I = identity(n, format='csr')
    A = I + alpha * L
    # Ensure CSR format for spsolve
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)
    # Initial labels: positive=1, negative=-1, unlabeled=0 (but we label all here)
    y = np.zeros(n)
    y[labels == 1] = 1.0
    y[labels == -1] = -1.0
    # Solve for f
    f = spsolve(A, y)
    # Convert to probability of positive class: p = (f + 1) / 2
    prob = (f + 1) / 2
    # Clip to [0,1]
    prob = np.clip(prob, 0, 1)
    return prob

def graph_laplacian_score(returns, labels, alpha=0.2, n_neighbors=5):
    """Wrapper for train.py: compute final probability for each ETF."""
    prob = label_propagation(returns, labels, alpha, n_neighbors)
    return prob

def generate_labels_from_future(returns_df, window_days):
    """Generate binary labels from the last day's return (self‑supervised)."""
    if len(returns_df) < window_days + 1:
        return None
    last_return = returns_df.iloc[-1].values
    labels = np.zeros(len(last_return))
    labels[last_return > 0] = 1
    labels[last_return < 0] = -1
    return labels
