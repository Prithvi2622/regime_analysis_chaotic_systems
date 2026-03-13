import numpy as np

try:
    import nolds
    HAS_NOLDS = True
except ImportError:
    HAS_NOLDS = False

def compute_largest_lyapunov_exponent(series: np.ndarray, m: int, tau: int = 1) -> float:
    """
    Computes the Largest Lyapunov Exponent (LLE) using Rosenstein's algorithm.
    Positive LLE is an indicator of chaos.
    """
    if not HAS_NOLDS:
        raise ImportError("nolds is required to compute Lyapunov exponent.")
        
    try:
        # lyap_r returns the largest lyapunov exponent
        lle = nolds.lyap_r(series, emb_dim=m, tau=tau)
        return float(lle)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error computing LLE: {e}")
        return np.nan
