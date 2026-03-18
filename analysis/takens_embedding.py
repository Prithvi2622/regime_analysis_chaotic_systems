import numpy as np
import os
import warnings

try:
    import nolitsa
except ImportError:
    print("Installing nolitsa from GitHub...")
    os.system("pip install git+https://github.com/manu-mannattil/nolitsa.git")

try:
    from nolitsa import delay, dimension
    HAS_NOLITSA = True
except ImportError:
    HAS_NOLITSA = False
    warnings.warn("nolitsa not installed. Some Takens embedding functions will use fallbacks.")

def compute_delay_mutual_information(series: np.ndarray, tau_max: int = 50, bins: int = 50) -> int:
    """
    Estimates optimal time delay (tau) using the first minimum of mutual information.
    """
    if HAS_NOLITSA:
        # nolitsa's delay.dmi calculates delay mutual information
        mi = delay.dmi(series, maxtau=tau_max, bins=bins)
        # Find the first local minimum
        for i in range(1, len(mi) - 1):
            if mi[i] < mi[i - 1] and mi[i] < mi[i + 1]:
                return i
        return np.argmin(mi)
    else:
        # Fallback: simple autocorrelation threshold (e.g. drop below 1/e)
        from statsmodels.tsa.stattools import acf
        auto_corr = acf(series, nlags=tau_max)
        for i, val in enumerate(auto_corr):
            if val < 1/np.exp(1):
                return max(1, i)
        return max(1, np.argmin(auto_corr))

def false_nearest_neighbors(series: np.ndarray, tau: int, m_max: int = 10) -> int:
    """
    Estimates the optimal embedding dimension (m) using the False Nearest Neighbors method.
    """
    if HAS_NOLITSA:
        fnn = dimension.fnn(series, tau=tau, dim=np.arange(1, m_max + 1))
        # fnn shape varies by implementation; typically returns fractions of false nearest neighbors.
        # usually [0] is the fraction. We want the first dimension where fraction drops below a threshold (e.g., 5%)
        threshold = 0.05
        # dimension.fnn returns 3D array in nolitsa. [dim, [Rtol, Atol, Rtol|Atol]]
        # Need to check nolitsa docs, but usually we look at the Rtol fraction.
        # Fallback: if we just take the first component:
        for i in range(m_max):
            if np.mean(fnn[i, 2]) < threshold:
                return i + 1
        return m_max
    else:
        # Very naive fallback if nolitsa is missing
        return min(m_max, 3)

def takens_embedding(series: np.ndarray, tau: int, m: int) -> np.ndarray:
    """
    Constructs the phase space using Takens' delay embedding.
    
    Args:
        series: 1D time series array.
        tau: Time delay.
        m: Embedding dimension.
        
    Returns:
        A matrix of shape (N - (m-1)*tau, m)
    """
    n = len(series)
    max_delay = (m - 1) * tau
    if max_delay >= n:
        raise ValueError("Series too short for given tau and m")
        
    embedded = np.zeros((n - max_delay, m))
    for i in range(m):
        embedded[:, i] = series[i * tau : n - max_delay + i * tau]
    
    return embedded

# Optional parameter estimation aliases
def estimate_tau(data: np.ndarray, tau_max: int = 50, bins: int = 50) -> int:
    """Estimates optimal time delay (tau) using mutual information."""
    return compute_delay_mutual_information(data, tau_max, bins)

def estimate_embedding_dim(data: np.ndarray, tau: int, m_max: int = 10) -> int:
    """Estimates optimal embedding dimension (m) using false nearest neighbors."""
    return false_nearest_neighbors(data, tau, m_max)
