import numpy as np

def compute_distances(points: np.ndarray, metric: str = 'euclidean'):
    """
    Compute pairwise distances between points in phase space.
    """
    from scipy.spatial.distance import pdist, squareform
    return squareform(pdist(points, metric=metric))

def system_entropy(probabilities: np.ndarray) -> float:
    """
    Compute Shannon entropy given probability distribution.
    """
    p = probabilities[probabilities > 0]
    return -np.sum(p * np.log2(p))
