import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import sys
    from unittest.mock import MagicMock
    sys.modules['nolds.datasets'] = MagicMock()
    import nolds
    HAS_NOLDS = True
except ImportError:
    HAS_NOLDS = False

def compute_correlation_dimension(series: np.ndarray, m: int) -> float:
    """
    Compute correlation dimension.
    """
    if not HAS_NOLDS:
        raise ImportError("nolds is required for correlation dimension.")
    
    # nolds.corr_dim expects a 1D array and calculates the dimension using Grassberger-Procaccia
    try:
        cd = nolds.corr_dim(series, emb_dim=m)
        return float(cd)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error computing corr_dim: {e}")
        return np.nan

def compute_sample_entropy(series: np.ndarray, m: int, r: float = None) -> float:
    """
    Compute sample entropy.
    """
    if not HAS_NOLDS:
        raise ImportError("nolds is required for sample entropy.")
        
    if r is None:
        r = 0.2 * np.std(series)
        
    try:
        se = nolds.sampen(series, emb_dim=m, tolerance=r)
        return float(se)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error computing sampen: {e}")
        return np.nan

def get_chaos_metrics(series: np.ndarray, m: int, from_pyunicorn: bool = False) -> dict:
    """
    Compile standard metrics.
    """
    metrics = {}
    
    if HAS_NOLDS:
        metrics['correlation_dimension'] = compute_correlation_dimension(series, m)
        metrics['sample_entropy'] = compute_sample_entropy(series, m)
        
    if from_pyunicorn:
        try:
            import pyunicorn.timeseries as ts
            # Create a recurrence plot object (requires embedded or 1D)
            rp = ts.RecurrencePlot(series.reshape(-1, 1), metric='euclidean', normalize=True)
            metrics['determinism'] = rp.determinism()
            metrics['laminarity'] = rp.laminarity()
        except ImportError:
            print("pyunicorn not available, skipping")
            metrics['pyunicorn_error'] = "not installed"
    
    return metrics
