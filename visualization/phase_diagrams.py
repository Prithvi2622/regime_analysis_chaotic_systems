import matplotlib.pyplot as plt
import numpy as np

def plot_regime_segmentation(time: np.ndarray, series: np.ndarray, regimes: np.ndarray,
                             title="Regime Segmentation Timeline", save_path=None):
    """
    Plots the time series with backgrounds colored by the detected regime.
    
    Args:
        time: 1D array of time points
        series: 1D array of the series values (e.g., prices or returns)
        regimes: 1D array of predicted regime labels (integers)
        title: Plot title
        save_path: Optional save path
    """
    # Create colors
    unique_regimes = np.unique(regimes)
    colors = plt.cm.get_cmap('tab10', len(unique_regimes))
    
    plt.figure(figsize=(12, 6))
    plt.plot(time, series, color='black', alpha=0.7, lw=1)
    
    # Shade backgrounds according to regime
    for r in unique_regimes:
        mask = (regimes == r)
        plt.fill_between(time, series.min(), series.max(), where=mask,
                         color=colors(r), alpha=0.3, label=f'Regime {r}')
                         
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    # Simplify legend by removing duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_changepoints(time: np.ndarray, series: np.ndarray, changepoints: list,
                      title="Change Point Detection Result", save_path=None):
    """
    Plots the time series with dashed vertical lines for detected changepoints.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time, series, color='blue', alpha=0.8, lw=1)
    
    for cp in changepoints[:-1]: # Ruptures usually appends the end index, don't plot it
        if cp < len(time):
            plt.axvline(time[cp], color='red', linestyle='--', alpha=0.8)
            
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    
    if save_path:
        plt.savefig(save_path)
    plt.close()
