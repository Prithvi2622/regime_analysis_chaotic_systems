import numpy as np
import pandas as pd
from scipy.signal import medfilt
# Note: For advanced wavelet denoising, pywt would be needed. 
# We implement a simple moving average / median filter for now if pywt is not available.
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

def compute_log_returns(df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
    """
    Computes log returns from price series.
    """
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in dataframe")
        
    # Calculate log returns: ln(P_t / P_{t-1})
    df['returns'] = np.log(df[price_col] / df[price_col].shift(1))
    
    # Fill the first NaN value
    df['returns'] = df['returns'].bfill()
    return df

def normalize_series(series: np.ndarray) -> np.ndarray:
    """
    Z-score normalization of a time series.
    """
    mean = np.mean(series)
    std = np.std(series)
    if std == 0:
        return series - mean
    return (series - mean) / std

def denoise_signal(series: np.ndarray, method: str = 'wavelet', **kwargs) -> np.ndarray:
    """
    Denoises a 1D signal.
    
    Args:
        series: numpy array of the signal
        method: 'wavelet', 'moving_average', or 'median'
        
    Returns:
        Denoised numpy array
    """
    if method == 'wavelet' and HAS_PYWT:
        wavelet = kwargs.get('wavelet_name', 'db4')
        level = kwargs.get('wavelet_level', 2)
        
        # Deconstruct the signal
        coeffs = pywt.wavedec(series, wavelet, level=level)
        
        # Calculate a threshold (universal threshold)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(series)))
        
        # Apply soft thresholding to detail coefficients
        coeffs[1:] = (pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:])
        
        # Reconstruct the signal
        return pywt.waverec(coeffs, wavelet)
        
    elif method == 'median' or (method == 'wavelet' and not HAS_PYWT):
        # Fallback or explicit median filter
        window = kwargs.get('window_size', 5)
        if window % 2 == 0:
            window += 1
        return medfilt(series, kernel_size=window)
        
    elif method == 'moving_average':
        window = kwargs.get('window_size', 5)
        weights = np.ones(window) / window
        return np.convolve(series, weights, mode='same')
        
    else:
        raise ValueError(f"Unknown denoising method: {method}")
