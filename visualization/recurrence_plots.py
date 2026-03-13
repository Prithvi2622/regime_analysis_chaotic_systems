import matplotlib.pyplot as plt
import numpy as np

def plot_recurrence_matrix(rm: np.ndarray, title="Recurrence Plot", save_path=None):
    """
    Visualizes a Recurrence Matrix.
    
    Args:
        rm: 2D numpy array representing the recurrence matrix.
        title: Title of the plot.
        save_path: Optional path to save the plot.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(rm, cmap='binary', origin='lower')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Time')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def compute_and_plot_recurrence(points: np.ndarray, threshold: float = 0.1, save_path=None):
    """
    Computes a simple recurrence plot and visualizes it.
    """
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(points, metric='euclidean'))
    rm = distances < threshold
    
    plot_recurrence_matrix(rm, save_path=save_path)
    return rm
