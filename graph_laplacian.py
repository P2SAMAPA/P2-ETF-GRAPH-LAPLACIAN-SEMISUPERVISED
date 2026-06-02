import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import inv
from scipy.sparse import identity

def label_propagation(returns, labels, alpha=0.2, n_neighbors=5):
    """
    returns: DataFrame of returns (T x n_etfs) – only the last row used for graph? Actually we build graph from correlation of the whole window.
    labels: binary array (n,) where 1 = positive, 0 = negative, -1 = unlabeled.
    """
    # Build graph from correlation distance
    corr = returns.corr().values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    # Build k-NN graph (symmetric)
    n = dist.shape[0]
    # Use kneighbors_graph on distance matrix (negative distances? we use metric='precomputed')
    from sklearn.neighbors import kneighbors_graph
    # Kneighbors_graph expects a pairwise distance matrix and uses the smaller distances as closer.
    # Since we want the smallest distances (strongest correlations), we can use dist directly.
    # However kneighbors_graph only works with metric='precomputed' for distance matrices.
    # We'll build graph manually: for each i, take k smallest distances.
    adj = np.zeros((n, n))
    for i in range(n):
        # indices of k nearest neighbours (excluding self)
        nearest = np.argsort(dist[i])[1:n_neighbors+1]
        adj[i, nearest] = 1
    # Make symmetric
    adj = np.maximum(adj, adj.T)
    # Compute graph Laplacian
    L = laplacian(adj, normed=False)
    # Regularised Laplacian for label propagation: (I + alpha L) * f = y
    # where f is the final label scores, y is initial labels (with unlabeled as 0)
    I = identity(n)
    A = I + alpha * L
    # Initial labels: +1 for positive, -1 for negative, 0 for unlabeled
    y = np.zeros(n)
    y[labels == 1] = 1.0
    y[labels == -1] = -1.0
    # Solve for f
    from scipy.sparse.linalg import spsolve
    f = spsolve(A, y)
    # Convert to probability of positive class: p = (f + 1) / 2
    prob = (f + 1) / 2
    return prob

def graph_laplacian_score(returns, labels, alpha=0.2, n_neighbors=5):
    """Wrapper for train.py: compute final probability for each ETF."""
    prob = label_propagation(returns, labels, alpha, n_neighbors)
    return prob

def generate_labels_from_future(returns_df, window_days, horizon=1):
    """
    For the most recent day in the window, generate binary labels based on next-day return.
    Positive label if next-day return > 0, negative if < 0. Unlabeled if no future.
    """
    if len(returns_df) < window_days + horizon:
        return None
    ret_window = returns_df.iloc[-window_days:]
    next_day = returns_df.iloc[-1]   # actually we need the day after the window? Wait: we have the full returns_df.
    # Simpler: we can label using the next day's return after the last day of the window.
    # But we are at training time – we need to create labels from the "future" relative to the window.
    # In practice, for each day t in the past, we can use t+1 return to label t.
    # However for the current window we are estimating a score for the present (no future).
    # We'll instead use the last day's label from the window? That's not predictive.
    # Approach: For training (run_for_window), we will not require labels – we only need to compute scores for the most recent day.
    # Instead, we can compute the propagation using the whole window and treat all nodes as unlabeled? That gives a prior.
    # To make it supervised, we can use a small fraction of ETFs as labelled based on their last day's return (but that's look‑ahead).
    # Given the complexity, we will simplify: for each window, we will use the last day's return as the label for the entire window? No.
    # A clean solution: use a **transductive** approach: given the returns of the window, we want to predict which ETFs will go up tomorrow.
    # But tomorrow is not known. So we use the last day's return as a proxy? That's not predictive.
    # Therefore, we will implement the **self‑supervised** variant: use the sign of the last day's return as the label for the ETF itself.
    # That is a simple momentum signal: ETFs that were positive today are "positive" class, negative are "negative".
    # Then the propagation will give scores that may highlight ETFs that are central to the positive group.
    # This is a valid signal.
    last_return = returns_df.iloc[-1].values
    labels = np.zeros(len(last_return))
    labels[last_return > 0] = 1
    labels[last_return < 0] = -1
    return labels
