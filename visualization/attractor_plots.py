import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

def plot_2d_attractor(trajectory: np.ndarray, title="2D Phase Space Attractor", save_path=None):
    """
    Plots a 2D projection of the attractor.
    """
    plt.figure(figsize=(8, 6))
    idx = np.arange(0, len(trajectory), step=10)
    embedded_plot = np.clip(trajectory[idx], -5, 5)
    if trajectory.shape[1] >= 2:
        plt.scatter(embedded_plot[:, 0], embedded_plot[:, 1], s=1, alpha=0.5)
    else:
        plt.plot(embedded_plot)
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_3d_attractor(trajectory: np.ndarray, title="3D Phase Space Attractor", save_path=None):
    """
    Plots a 3D projection of the attractor using matplotlib.
    """
    if trajectory.shape[1] < 3:
        raise ValueError("Trajectory must have at least 3 dimensions for a 3D plot")
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    idx = np.arange(0, len(trajectory), step=10)
    embedded_plot = np.clip(trajectory[idx], -5, 5)
    ax.scatter(embedded_plot[:, 0], embedded_plot[:, 1], embedded_plot[:, 2], s=1, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_3d_attractor_interactive(trajectory: np.ndarray, title="Interactive 3D Attractor", save_path=None):
    """
    Plots an interactive 3D attractor using Plotly.
    """
    if trajectory.shape[1] < 3:
        raise ValueError("Trajectory must have at least 3 dimensions for a 3D plot")
        
    fig = go.Figure(data=[go.Scatter3d(
        x=trajectory[:, 0],
        y=trajectory[:, 1],
        z=trajectory[:, 2],
        mode='lines',
        line=dict(width=2, color=trajectory[:, 2], colorscale='Viridis')
    )])
    
    fig.update_layout(title=title, margin=dict(l=0, r=0, b=0, t=30))
    if save_path:
        fig.write_html(save_path)
    return fig
