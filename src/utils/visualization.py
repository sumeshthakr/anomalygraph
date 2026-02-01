"""
Visualization Utilities for Point Cloud Anomaly Detection

Provides functions for visualizing:
1. Point cloud anomaly heatmaps
2. Original point clouds
3. Comparison views
"""

import numpy as np
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def visualize_heatmap(
    points: np.ndarray,
    heatmap: np.ndarray,
    title: str = "Anomaly Heatmap",
    colormap: str = "jet",
    point_size: float = 1.0,
    figsize: Tuple[int, int] = (10, 8),
    elevation: float = 20,
    azimuth: float = 45,
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Visualize point cloud with anomaly heatmap overlay.
    
    Args:
        points: Point coordinates (N, 3)
        heatmap: Per-point anomaly scores (N,), normalized to [0, 1]
        title: Plot title
        colormap: Matplotlib colormap name
        point_size: Size of points
        figsize: Figure size
        elevation: View elevation angle
        azimuth: View azimuth angle
        save_path: Path to save figure (optional)
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure if show=False
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize heatmap if needed
    if heatmap.max() > 1.0 or heatmap.min() < 0.0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Get colormap
    cmap = cm.get_cmap(colormap)
    colors = cmap(heatmap)
    
    # Plot points
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=heatmap,
        cmap=colormap,
        s=point_size,
        alpha=0.8
    )
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('Anomaly Score')
    
    # Set view
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Equal aspect ratio
    _set_axes_equal(ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
        return None
    else:
        return fig


def visualize_point_cloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    title: str = "Point Cloud",
    point_size: float = 1.0,
    figsize: Tuple[int, int] = (10, 8),
    elevation: float = 20,
    azimuth: float = 45,
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Visualize a 3D point cloud.
    
    Args:
        points: Point coordinates (N, 3)
        colors: Point colors (N, 3) or (N,) for grayscale
        title: Plot title
        point_size: Size of points
        figsize: Figure size
        elevation: View elevation angle
        azimuth: View azimuth angle
        save_path: Path to save figure
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure if show=False
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if colors is None:
        # Default to height-based coloring
        colors = points[:, 2]
    
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=point_size,
        alpha=0.8
    )
    
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    _set_axes_equal(ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
        return None
    else:
        return fig


def visualize_comparison(
    points: np.ndarray,
    heatmap: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    title: str = "Anomaly Detection Result",
    figsize: Tuple[int, int] = (18, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Visualize comparison of original, prediction, and ground truth.
    
    Args:
        points: Point coordinates (N, 3)
        heatmap: Predicted anomaly heatmap (N,)
        ground_truth: Ground truth labels (N,), optional
        title: Overall title
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure if show=False
    """
    n_plots = 3 if ground_truth is not None else 2
    fig = plt.figure(figsize=figsize)
    
    # Original point cloud
    ax1 = fig.add_subplot(1, n_plots, 1, projection='3d')
    ax1.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=points[:, 2], s=0.5, alpha=0.8
    )
    ax1.set_title('Original')
    _set_axes_equal(ax1)
    
    # Predicted heatmap
    ax2 = fig.add_subplot(1, n_plots, 2, projection='3d')
    scatter = ax2.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=heatmap, cmap='jet', s=0.5, alpha=0.8
    )
    ax2.set_title('Predicted Anomaly')
    fig.colorbar(scatter, ax=ax2, shrink=0.6)
    _set_axes_equal(ax2)
    
    # Ground truth if available
    if ground_truth is not None:
        ax3 = fig.add_subplot(1, n_plots, 3, projection='3d')
        ax3.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=ground_truth, cmap='jet', s=0.5, alpha=0.8
        )
        ax3.set_title('Ground Truth')
        _set_axes_equal(ax3)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
        return None
    else:
        return fig


def visualize_multi_view(
    points: np.ndarray,
    heatmap: np.ndarray,
    views: List[Tuple[float, float]] = [(0, 0), (0, 90), (90, 0), (45, 45)],
    title: str = "Multi-View Anomaly Heatmap",
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Visualize point cloud from multiple viewpoints.
    
    Args:
        points: Point coordinates (N, 3)
        heatmap: Anomaly heatmap (N,)
        views: List of (elevation, azimuth) tuples
        title: Overall title
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure if show=False
    """
    n_views = len(views)
    rows = int(np.ceil(np.sqrt(n_views)))
    cols = int(np.ceil(n_views / rows))
    
    fig = plt.figure(figsize=figsize)
    
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        
        scatter = ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=heatmap, cmap='jet', s=0.5, alpha=0.8
        )
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'View ({elev}°, {azim}°)')
        _set_axes_equal(ax)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
        return None
    else:
        return fig


def _set_axes_equal(ax):
    """Set equal aspect ratio for 3D axes."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])
    
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def create_plotly_heatmap(
    points: np.ndarray,
    heatmap: np.ndarray,
    title: str = "Anomaly Heatmap",
    point_size: float = 2.0,
    colorscale: str = "Jet"
) -> "go.Figure":
    """
    Create interactive 3D visualization using Plotly.
    
    Args:
        points: Point coordinates (N, 3)
        heatmap: Anomaly scores (N,)
        title: Plot title
        point_size: Size of points
        colorscale: Plotly colorscale name
        
    Returns:
        Plotly Figure object
    """
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=heatmap,
                colorscale=colorscale,
                colorbar=dict(title='Anomaly Score'),
                opacity=0.8
            ),
            text=[f'Score: {s:.4f}' for s in heatmap],
            hoverinfo='text'
        )
    ])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        )
    )
    
    return fig


def save_plotly_html(
    points: np.ndarray,
    heatmap: np.ndarray,
    save_path: str,
    title: str = "Anomaly Heatmap"
):
    """Save interactive Plotly visualization to HTML file."""
    fig = create_plotly_heatmap(points, heatmap, title)
    fig.write_html(save_path)
    print(f"Saved interactive visualization to {save_path}")


class TrainingVisualizer:
    """
    Visualization utilities for training progress.
    """
    
    def __init__(self, save_dir: str = './visualizations'):
        self.save_dir = save_dir
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
    
    def log(
        self,
        train_loss: float,
        val_loss: Optional[float] = None,
        lr: Optional[float] = None
    ):
        """Log training progress."""
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if lr is not None:
            self.learning_rates.append(lr)
    
    def plot_losses(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """Plot training and validation losses."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        ax = axes[0]
        ax.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            ax.plot(self.val_losses, label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate
        ax = axes[1]
        if self.learning_rates:
            ax.plot(self.learning_rates)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        
        if show:
            plt.show()
    
    def plot_metrics(
        self,
        metrics: dict,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """Plot evaluation metrics."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        bars = ax.bar(categories, values)
        ax.set_ylabel('AUROC')
        ax.set_title('Per-Category AUROC')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{val:.3f}',
                ha='center'
            )
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        
        if show:
            plt.show()
