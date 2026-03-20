import numpy as np

try:
    import pysindy as ps
    HAS_PYSINDY = True
except ImportError:
    HAS_PYSINDY = False

def fit_sparse_dynamics(x: np.ndarray, t: np.ndarray, 
                        poly_degree: int = 3, threshold: float = 0.01, alpha: float = 0.05):
    """
    Fit a SINDy model to discover governing equations.
    
    Args:
        x: data matrix of shape (n_samples, n_features)
        t: time steps (either scalar dt or array of time values)
        poly_degree: Degree of polynomial library.
        threshold: STLSQ threshold.
        alpha: STLSQ alpha (ridge penalty).
        
    Returns:
        Fitted PySINDy model.
    """
    if not HAS_PYSINDY:
        raise ImportError("pysindy is required for SINDy dynamics discovery.")
        
    # Feature library
    library = ps.PolynomialLibrary(degree=poly_degree)
    
    # Optimizer (Sparse Relaxed Regularized Regression or STLSQ)
    optimizer = ps.STLSQ(threshold=threshold, alpha=alpha)
    
    # Instantiate and fit
    model = ps.SINDy(feature_library=library, optimizer=optimizer)
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Fitting SINDy model with threshold={threshold}, degree={poly_degree}, dt={t}")
    if isinstance(t, (float, int)):
        model.fit(x, t=float(t))
    else:
        model.fit(x, t=t)
        
    return model

def discover_equations(model):
    """
    Print the discovered governing equations.
    """
    model.print()
    return model.equations()

def simulate_system(model, x0: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Simulate the system forward in time using the learned SINDy model.
    
    Args:
        model: Fitted PySINDy model.
        x0: Initial condition (1D array of shape (n_features,)).
        t: Array of time points to simulate over.
        
    Returns:
        Simulated trajectory of shape (n_samples, n_features).
    """
    return model.simulate(x0, t)
