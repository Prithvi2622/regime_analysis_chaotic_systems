import numpy as np
try:
    from hmmlearn import hmm
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False

def train_hmm(data: np.ndarray, n_components: int = 3, covariance_type: str = "full", n_iter: int = 100):
    """
    Trains a Gaussian Hidden Markov Model.
    
    Args:
        data: shape (n_samples, n_features)
        n_components: Number of hidden states (regimes).
        covariance_type: 'spherical', 'tied', 'diag', 'full'.
        n_iter: Max iterations for EM algorithm.
        
    Returns:
        Fitted HMM model.
    """
    if not HAS_HMMLEARN:
        raise ImportError("hmmlearn is required for HMM regime detection.")
        
    model = hmm.GaussianHMM(
        n_components=n_components, 
        covariance_type=covariance_type, 
        n_iter=n_iter,
        random_state=42
    )
    
    model.fit(data)
    return model

def predict_regimes(model, data: np.ndarray):
    """
    Predicts regimes using a trained HMM.
    
    Args:
        model: Trained hmmlearn model.
        data: shape (n_samples, n_features).
        
    Returns:
        tuple: (regime labels, regime probabilities)
    """
    labels = model.predict(data)
    probabilities = model.predict_proba(data)
    
    return labels, probabilities
