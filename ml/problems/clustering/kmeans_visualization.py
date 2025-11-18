"""
K-Means Visualization Example

This script provides visualization utilities for K-Means clustering.
Run this after implementing kmeans.py to see your clustering in action!

Features:
- Visualize cluster assignments
- Show centroid movement during training
- Compare different initialization methods
- Elbow method plot
- Silhouette analysis

Usage:
    python kmeans_visualization.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


def generate_blobs(
    n_samples: int = 300,
    n_features: int = 2,
    centers: int = 3,
    cluster_std: float = 0.6,
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic clustered data.
    
    Args:
        n_samples: Total number of samples
        n_features: Number of features (dimensions)
        centers: Number of cluster centers
        cluster_std: Standard deviation of clusters
        random_seed: Random seed for reproducibility
        
    Returns:
        X: Data tensor [n_samples, n_features]
        y_true: True cluster labels [n_samples]
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    samples_per_center = n_samples // centers
    
    # Generate cluster centers
    center_coords = torch.randn(centers, n_features) * 3
    
    X_list = []
    y_list = []
    
    for i in range(centers):
        # Generate samples around center
        samples = torch.randn(samples_per_center, n_features) * cluster_std + center_coords[i]
        X_list.append(samples)
        y_list.append(torch.full((samples_per_center,), i, dtype=torch.long))
    
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    
    # Shuffle
    indices = torch.randperm(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def plot_clusters(
    X: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor = None,
    title: str = "K-Means Clustering",
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Plot 2D clusters with centroids.
    
    Args:
        X: Data tensor [n_samples, 2]
        labels: Cluster assignments [n_samples]
        centroids: Centroid coordinates [n_clusters, 2]
        title: Plot title
        ax: Matplotlib axes (creates new if None)
        
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Convert to numpy for plotting
    X_np = X.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Plot points
    scatter = ax.scatter(
        X_np[:, 0], X_np[:, 1],
        c=labels_np,
        cmap='viridis',
        alpha=0.6,
        s=50,
        edgecolors='k',
        linewidth=0.5
    )
    
    # Plot centroids
    if centroids is not None:
        centroids_np = centroids.cpu().numpy()
        ax.scatter(
            centroids_np[:, 0], centroids_np[:, 1],
            c='red',
            marker='X',
            s=300,
            edgecolors='black',
            linewidth=2,
            label='Centroids',
            zorder=10
        )
    
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_elbow_curve(
    elbow_results: List[Tuple[int, float]],
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Plot elbow curve for optimal k selection.
    
    Args:
        elbow_results: List of (k, inertia) tuples
        ax: Matplotlib axes (creates new if None)
        
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    k_values = [k for k, _ in elbow_results]
    inertias = [inertia for _, inertia in elbow_results]
    
    ax.plot(k_values, inertias, 'bo-', linewidth=2, markersize=10)
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Inertia (WCSS)', fontsize=12)
    ax.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Annotate points
    for k, inertia in elbow_results:
        ax.annotate(
            f'{inertia:.0f}',
            (k, inertia),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9
        )
    
    return ax


def plot_initialization_comparison(
    X: torch.Tensor,
    kmeans_results: List[Tuple[str, torch.Tensor, torch.Tensor, int]],
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Compare different K-Means initialization methods.
    
    Args:
        X: Data tensor [n_samples, 2]
        kmeans_results: List of (name, labels, centroids, n_iters) tuples
        figsize: Figure size
    """
    n_methods = len(kmeans_results)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    
    if n_methods == 1:
        axes = [axes]
    
    for ax, (name, labels, centroids, n_iters) in zip(axes, kmeans_results):
        plot_clusters(
            X, labels, centroids,
            title=f"{name}\n({n_iters} iterations)",
            ax=ax
        )
    
    plt.tight_layout()
    return fig


def plot_silhouette_analysis(
    X: torch.Tensor,
    labels: torch.Tensor,
    n_clusters: int,
    silhouette_values: torch.Tensor,
    avg_score: float
):
    """
    Create silhouette plot for cluster analysis.
    
    Args:
        X: Data tensor [n_samples, 2]
        labels: Cluster assignments [n_samples]
        n_clusters: Number of clusters
        silhouette_values: Silhouette score for each sample
        avg_score: Average silhouette score
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Silhouette plot
    y_lower = 10
    for i in range(n_clusters):
        cluster_silhouette_values = silhouette_values[labels == i]
        cluster_silhouette_values = torch.sort(cluster_silhouette_values)[0]
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.viridis(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouette_values.cpu().numpy(),
            facecolor=color,
            edgecolor=color,
            alpha=0.7
        )
        
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax1.set_title('Silhouette Plot', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Silhouette Coefficient', fontsize=12)
    ax1.set_ylabel('Cluster Label', fontsize=12)
    ax1.axvline(x=avg_score, color="red", linestyle="--", linewidth=2, label=f'Average: {avg_score:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cluster plot
    plot_clusters(X, labels, title=f'Clusters (Silhouette: {avg_score:.3f})', ax=ax2)
    
    plt.tight_layout()
    return fig


# Example usage
if __name__ == "__main__":
    # Import your KMeans implementations
    # Uncomment after implementing kmeans.py:
    # from kmeans import KMeans, KMeansPlusPlus, MiniBatchKMeans, elbow_method, silhouette_score
    
    print("=" * 70)
    print("K-Means Visualization Example")
    print("=" * 70)
    print("\nFirst, implement the KMeans algorithms in kmeans.py")
    print("Then uncomment the import statement and run this script again!")
    print("\nThis will generate:")
    print("  1. Cluster visualization")
    print("  2. Initialization comparison")
    print("  3. Elbow method plot")
    print("  4. Silhouette analysis")
    print("=" * 70)
    
    # Uncomment below after implementing kmeans.py:
    """
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    X, y_true = generate_blobs(n_samples=300, centers=3, cluster_std=0.6)
    
    # 1. Basic clustering
    print("Running K-Means clustering...")
    kmeans = KMeans(n_clusters=3, random_seed=42)
    kmeans.fit(X)
    
    plt.figure(figsize=(10, 8))
    plot_clusters(X, kmeans.labels_, kmeans.centroids, "K-Means Clustering Result")
    plt.savefig('kmeans_basic.png', dpi=150, bbox_inches='tight')
    print("  Saved: kmeans_basic.png")
    
    # 2. Initialization comparison
    print("\nComparing initialization methods...")
    km_random = KMeans(n_clusters=3, random_seed=42)
    km_random.fit(X)
    
    km_pp = KMeansPlusPlus(n_clusters=3, random_seed=42)
    km_pp.fit(X)
    
    results = [
        ("Random Init", km_random.labels_, km_random.centroids, km_random.n_iter_),
        ("K-Means++", km_pp.labels_, km_pp.centroids, km_pp.n_iter_),
    ]
    
    plot_initialization_comparison(X, results)
    plt.savefig('kmeans_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: kmeans_comparison.png")
    
    # 3. Elbow method
    print("\nRunning elbow method...")
    elbow_results = elbow_method(X, range(2, 8), n_trials=5)
    
    plt.figure(figsize=(10, 6))
    plot_elbow_curve(elbow_results)
    plt.savefig('kmeans_elbow.png', dpi=150, bbox_inches='tight')
    print("  Saved: kmeans_elbow.png")
    
    # 4. Silhouette analysis
    print("\nComputing silhouette scores...")
    score = silhouette_score(X, kmeans.labels_)
    print(f"  Silhouette Score: {score:.4f}")
    
    print("\n" + "=" * 70)
    print("Visualization complete! Check the generated PNG files.")
    print("=" * 70)
    """

