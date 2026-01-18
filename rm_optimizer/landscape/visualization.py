"""
Visualization utilities for loss landscape analysis.

Provides:
- Eigenvalue distribution plots
- 2D loss surface projections
- Optimizer comparison charts
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Any
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_eigenvalue_distribution(
    eigenvalues: np.ndarray,
    title: str = "Hessian Eigenvalue Distribution",
    save_path: Optional[str] = None,
    log_scale: bool = True,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plot eigenvalue distribution.
    
    Args:
        eigenvalues: Array of eigenvalues (sorted descending)
        title: Plot title
        save_path: Path to save figure
        log_scale: Use log scale for y-axis
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Bar plot of top eigenvalues
    ax1 = axes[0]
    x = np.arange(len(eigenvalues))
    ax1.bar(x, eigenvalues, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Eigenvalue Index')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Top Eigenvalues')
    if log_scale and eigenvalues.min() > 0:
        ax1.set_yscale('log')
    
    # Right: Cumulative distribution
    ax2 = axes[1]
    cumsum = np.cumsum(eigenvalues) / eigenvalues.sum()
    ax2.plot(x, cumsum, 'o-', color='coral', linewidth=2, markersize=4)
    ax2.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% variance')
    ax2.set_xlabel('Number of Eigenvalues')
    ax2.set_ylabel('Cumulative Variance Explained')
    ax2.set_title('Variance Explained')
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_optimizer_comparison(
    results: Dict[str, List[Dict[str, float]]],
    metric: str = "val_accuracy",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plot comparison of optimizers.
    
    Args:
        results: Dict mapping optimizer name to list of result dicts
        metric: Metric to compare
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    optimizer_names = list(results.keys())
    positions = np.arange(len(optimizer_names))
    
    # Extract metric values for each optimizer
    means = []
    stds = []
    all_values = []
    
    for opt_name in optimizer_names:
        values = [r.get(metric, 0) for r in results[opt_name]]
        means.append(np.mean(values))
        stds.append(np.std(values))
        all_values.append(values)
    
    # Bar plot with error bars
    bars = ax.bar(
        positions, means,
        yerr=stds,
        capsize=5,
        color=sns.color_palette("husl", len(optimizer_names)),
        alpha=0.8,
        edgecolor='black',
        linewidth=1
    )
    
    # Add individual points
    for i, values in enumerate(all_values):
        x_jitter = positions[i] + np.random.normal(0, 0.05, len(values))
        ax.scatter(x_jitter, values, color='black', s=30, alpha=0.6, zorder=5)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(optimizer_names, rotation=45, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title or f'Optimizer Comparison: {metric}')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_loss_surface(
    model,
    dataloader,
    direction1: Optional[np.ndarray] = None,
    direction2: Optional[np.ndarray] = None,
    range_val: float = 1.0,
    num_points: int = 21,
    title: str = "Loss Surface",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
) -> plt.Figure:
    """
    Plot 2D loss surface projection.
    
    Projects the loss surface onto two random directions
    to visualize local geometry around the current minimum.
    
    Args:
        model: Trained model
        dataloader: Data for loss computation
        direction1: First direction vector (random if None)
        direction2: Second direction vector (random if None)
        range_val: Range of exploration in each direction
        num_points: Number of points per dimension
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    import torch
    
    device = next(model.parameters()).device
    
    # Get current parameters
    params = [p.data.clone() for p in model.parameters()]
    
    # Generate random directions if not provided
    if direction1 is None:
        direction1 = [torch.randn_like(p) for p in params]
        # Normalize
        norm1 = torch.sqrt(sum((d ** 2).sum() for d in direction1))
        direction1 = [d / norm1 for d in direction1]
    
    if direction2 is None:
        direction2 = [torch.randn_like(p) for p in params]
        # Orthogonalize to direction1
        proj = sum((d1 * d2).sum() for d1, d2 in zip(direction1, direction2))
        direction2 = [d2 - proj * d1 for d1, d2 in zip(direction1, direction2)]
        # Normalize
        norm2 = torch.sqrt(sum((d ** 2).sum() for d in direction2))
        direction2 = [d / norm2 for d in direction2]
    
    # Create grid
    alphas = np.linspace(-range_val, range_val, num_points)
    betas = np.linspace(-range_val, range_val, num_points)
    losses = np.zeros((num_points, num_points))
    
    print(f"Computing loss surface ({num_points}x{num_points} grid)...")
    
    model.eval()
    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                # Set parameters
                for p, p0, d1, d2 in zip(model.parameters(), params, direction1, direction2):
                    p.data = p0 + alpha * d1 + beta * d2
                
                # Compute loss
                total_loss = 0
                count = 0
                for batch in dataloader:
                    loss = model.compute_loss(
                        batch['chosen_input_ids'].to(device),
                        batch['chosen_attention_mask'].to(device),
                        batch['rejected_input_ids'].to(device),
                        batch['rejected_attention_mask'].to(device)
                    )
                    total_loss += loss.item()
                    count += 1
                    if count >= 5:  # Limit batches for speed
                        break
                
                losses[i, j] = total_loss / count
            
            if (i + 1) % 5 == 0:
                print(f"  Row {i+1}/{num_points}")
    
    # Restore original parameters
    for p, p0 in zip(model.parameters(), params):
        p.data = p0
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Contour plot
    ax1 = axes[0]
    X, Y = np.meshgrid(alphas, betas)
    contour = ax1.contourf(X, Y, losses.T, levels=50, cmap='viridis')
    ax1.contour(X, Y, losses.T, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    ax1.plot(0, 0, 'r*', markersize=15, label='Minimum')
    ax1.set_xlabel('Direction 1')
    ax1.set_ylabel('Direction 2')
    ax1.set_title('Loss Contours')
    ax1.legend()
    plt.colorbar(contour, ax=ax1, label='Loss')
    
    # 3D surface
    ax2 = axes[1]
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X, Y, losses.T, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('Direction 1')
    ax2.set_ylabel('Direction 2')
    ax2.set_zlabel('Loss')
    ax2.set_title('Loss Surface')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    histories: Dict[str, Dict[str, List[float]]],
    metrics: List[str] = None,
    title: str = "Training Curves",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4)
) -> plt.Figure:
    """
    Plot training curves for multiple optimizers.
    
    Args:
        histories: Dict mapping optimizer name to metric histories
        metrics: List of metrics to plot
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    metrics = metrics or ['train_loss', 'val_accuracy']
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    colors = sns.color_palette("husl", len(histories))
    
    for ax, metric in zip(axes, metrics):
        for (opt_name, history), color in zip(histories.items(), colors):
            if metric in history:
                values = history[metric]
                epochs = np.arange(1, len(values) + 1)
                ax.plot(epochs, values, label=opt_name, color=color, linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
