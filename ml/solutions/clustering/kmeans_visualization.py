"""
K-Means Visualization Helper

This script provides visualization utilities for K-Means clustering,
including interactive animations showing the algorithm's convergence.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from kmeans import KMeans


def plot_clusters(X, labels, centroids, title="K-Means Clustering"):
    """
    Plot data points colored by cluster assignment with centroids.
    
    Args:
        X: Data matrix of shape [n_samples, 2]
        labels: Cluster labels
        centroids: Centroid positions
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    
    # Plot points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                         alpha=0.6, edgecolors='k', s=50)
    
    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], 
               c='red', marker='X', s=300, edgecolors='black', 
               linewidths=2, label='Centroids')
    
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_kmeans_steps(X, n_clusters=3, max_iters=10, random_state=42):
    """
    Create an animated visualization of K-Means convergence.
    
    Args:
        X: Data matrix of shape [n_samples, 2]
        n_clusters: Number of clusters
        max_iters: Maximum iterations to show
        random_state: Random seed
    """
    # Initialize K-Means
    np.random.seed(random_state)
    kmeans = KMeans(n_clusters=n_clusters, max_iters=1, random_state=random_state, n_init=1)
    
    # Get initial centroids
    centroids = kmeans._initialize_centroids(X)
    
    # Track history
    centroid_history = [centroids.copy()]
    label_history = []
    
    # Run K-Means and record each step
    for _ in range(max_iters):
        labels = kmeans._assign_clusters(X, centroids)
        label_history.append(labels.copy())
        
        new_centroids = kmeans._update_centroids(X, labels)
        centroid_history.append(new_centroids.copy())
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            centroids = new_centroids
            break
        
        centroids = new_centroids
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame):
        ax.clear()
        
        if frame < len(label_history):
            labels = label_history[frame]
            centroids = centroid_history[frame]
            
            # Plot points
            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                               alpha=0.6, edgecolors='k', s=50)
            
            # Plot centroids
            ax.scatter(centroids[:, 0], centroids[:, 1], 
                      c='red', marker='X', s=300, edgecolors='black', 
                      linewidths=2, label='Centroids')
            
            # Plot centroid movement
            if frame > 0:
                prev_centroids = centroid_history[frame - 1]
                for i in range(n_clusters):
                    ax.arrow(prev_centroids[i, 0], prev_centroids[i, 1],
                           centroids[i, 0] - prev_centroids[i, 0],
                           centroids[i, 1] - prev_centroids[i, 1],
                           head_width=0.3, head_length=0.2, fc='red', ec='red', alpha=0.5)
            
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title(f'K-Means Iteration {frame + 1}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    anim = FuncAnimation(fig, update, frames=len(label_history), interval=1000, repeat=True)
    plt.tight_layout()
    plt.show()
    
    return anim


def compare_initializations(X, n_clusters=3, random_state=42):
    """
    Compare random initialization vs K-Means++ initialization.
    
    Args:
        X: Data matrix
        n_clusters: Number of clusters
        random_state: Random seed
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Random initialization
    kmeans_random = KMeans(n_clusters=n_clusters, init='random', 
                          random_state=random_state, n_init=1)
    labels_random = kmeans_random.fit_predict(X)
    
    axes[0].scatter(X[:, 0], X[:, 1], c=labels_random, cmap='viridis', 
                   alpha=0.6, edgecolors='k', s=50)
    axes[0].scatter(kmeans_random.centroids[:, 0], kmeans_random.centroids[:, 1],
                   c='red', marker='X', s=300, edgecolors='black', linewidths=2)
    axes[0].set_title(f'Random Initialization\nInertia: {kmeans_random.inertia_:.2f}\nIterations: {kmeans_random.n_iter_}')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].grid(True, alpha=0.3)
    
    # K-Means++ initialization
    kmeans_pp = KMeans(n_clusters=n_clusters, init='kmeans++', 
                      random_state=random_state, n_init=1)
    labels_pp = kmeans_pp.fit_predict(X)
    
    axes[1].scatter(X[:, 0], X[:, 1], c=labels_pp, cmap='viridis', 
                   alpha=0.6, edgecolors='k', s=50)
    axes[1].scatter(kmeans_pp.centroids[:, 0], kmeans_pp.centroids[:, 1],
                   c='red', marker='X', s=300, edgecolors='black', linewidths=2)
    axes[1].set_title(f'K-Means++ Initialization\nInertia: {kmeans_pp.inertia_:.2f}\nIterations: {kmeans_pp.n_iter_}')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Random init - Inertia: {kmeans_random.inertia_:.2f}, Iterations: {kmeans_random.n_iter_}")
    print(f"K-Means++ - Inertia: {kmeans_pp.inertia_:.2f}, Iterations: {kmeans_pp.n_iter_}")
    print(f"Improvement: {(1 - kmeans_pp.inertia_/kmeans_random.inertia_)*100:.2f}%")


def elbow_method(X, max_k=10, random_state=42):
    """
    Plot the elbow curve to help determine optimal K.
    
    Args:
        X: Data matrix
        max_k: Maximum number of clusters to try
        random_state: Random seed
    """
    inertias = []
    K_range = range(1, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    plt.title('Elbow Method for Optimal K', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("K-Means Visualization Examples")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    cluster1 = np.random.randn(100, 2) + np.array([0, 0])
    cluster2 = np.random.randn(100, 2) + np.array([5, 5])
    cluster3 = np.random.randn(100, 2) + np.array([10, 0])
    X = np.vstack([cluster1, cluster2, cluster3])
    
    print("\n1. Basic K-Means Clustering")
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    plot_clusters(X, labels, kmeans.centroids)
    
    print("\n2. Comparing Initialization Methods")
    compare_initializations(X, n_clusters=3, random_state=42)
    
    print("\n3. Elbow Method for Optimal K")
    elbow_method(X, max_k=10, random_state=42)
    
    print("\n4. Animated K-Means Convergence")
    print("   (Creating animation...)")
    # visualize_kmeans_steps(X, n_clusters=3, max_iters=10, random_state=42)
    print("   Note: Uncomment the line above to see the animation")
    
    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)

