import numpy as np

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False

def detect_changepoints_pelt(series: np.ndarray, model_type: str = "l2", pen: float = 10):
    """
    Detect change points using PELT (Pruned Exact Linear Time) algorithm.
    
    Args:
        series: 1D or 2D numpy array (n_samples, n_features)
        model_type: Cost function ("l1", "l2", "rbf").
        pen: Penalty value.
        
    Returns:
        List of indices of change points.
    """
    if not HAS_RUPTURES:
        raise ImportError("ruptures package is required for change point detection.")
        
    algo = rpt.Pelt(model=model_type).fit(series)
    result = algo.predict(pen=pen)
    return result

def detect_changepoints_binseg(series: np.ndarray, model_type: str = "l2", n_bkps: int = 5):
    """
    Detect change points using Binary Segmentation algorithm.
    
    Args:
        series: 1D or 2D numpy array.
        model_type: Cost function ("l1", "l2", "rbf", etc.).
        n_bkps: Number of change points to find.
        
    Returns:
        List of indices of change points.
    """
    if not HAS_RUPTURES:
        raise ImportError("ruptures package is required for change point detection.")
        
    algo = rpt.Binseg(model=model_type).fit(series)
    result = algo.predict(n_bkps=n_bkps)
    return result
