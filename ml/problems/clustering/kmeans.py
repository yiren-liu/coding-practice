"""
ML Implementation: K-Means Clustering and Variants

Description:
Implement K-Means clustering algorithm and its important variants:

1. Standard K-Means:
   - Lloyd's algorithm (expectation-maximization)
   - Random initialization
   - Iterative refinement until convergence

2. K-Means++ Initialization:
   - Smart initialization for better convergence
   - Probabilistic selection of initial centroids
   - Significantly improves clustering quality

3. Mini-Batch K-Means:
   - Faster variant for large datasets
   - Uses random mini-batches for updates
   - Better scalability with minimal quality loss

K-Means is one of the most widely used clustering algorithms in ML for:
- Customer segmentation
- Image compression and quantization
- Feature learning and preprocessing
- Anomaly detection

Mathematical Foundation:
- Objective: Minimize within-cluster sum of squares (WCSS)
- WCSS = Σ(i=1 to k) Σ(x in C_i) ||x - μ_i||²
- Where μ_i is the centroid of cluster C_i

References:
- Original K-Means: Lloyd (1957, 1982)
- K-Means++: Arthur & Vassilvitskii (2007) https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
- Mini-Batch: Sculley (2010) https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
import math


class KMeans:
    """
    Standard K-Means clustering using Lloyd's algorithm.
    
    Algorithm:
    1. Initialize k centroids randomly
    2. Repeat until convergence:
       a. Assign each point to nearest centroid (E-step)
       b. Update centroids as mean of assigned points (M-step)
    3. Stop when centroids don't change (or max iterations reached)
    
    Args:
        n_clusters: Number of clusters (k)
        max_iters: Maximum number of iterations
        tol: Tolerance for convergence (if centroid movement < tol, stop)
        random_seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_clusters: int = 8,
        max_iters: int = 300,
        tol: float = 1e-4,
        random_seed: Optional[int] = None
    ):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_seed = random_seed
        
        # Will be set during fit
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None  # Sum of squared distances to closest centroid
        self.n_iter_ = 0
    
    def fit(self, X: torch.Tensor) -> 'KMeans':
        """
        Fit K-Means to data.
        
        Args:
            X: Data tensor of shape [n_samples, n_features]
            
        Returns:
            self
        """
        # TODO: Implement K-Means fitting
        # 1. Initialize centroids randomly from data points
        #    Hint: Use torch.randperm for random sampling
        # 
        # 2. Iterate until convergence or max_iters:
        #    a. Assign labels (call _assign_labels)
        #    b. Update centroids (call _update_centroids)
        #    c. Check convergence (centroid movement < tol)
        # 
        # 3. Set self.labels_, self.inertia_, self.n_iter_
        pass
    
    def _assign_labels(self, X: torch.Tensor) -> torch.Tensor:
        """
        Assign each point to nearest centroid.
        
        Args:
            X: Data tensor of shape [n_samples, n_features]
            
        Returns:
            Labels tensor of shape [n_samples] with values in [0, k-1]
        """
        # TODO: Implement label assignment
        # 1. Compute distances from each point to each centroid
        #    Distance: ||x - c||² = ||x||² + ||c||² - 2(x·c)
        #    Or use torch.cdist for Euclidean distance
        # 
        # 2. Assign each point to nearest centroid
        #    Hint: Use torch.argmin along centroid dimension
        pass
    
    def _update_centroids(self, X: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Update centroids as mean of assigned points.
        
        Args:
            X: Data tensor of shape [n_samples, n_features]
            labels: Current cluster assignments [n_samples]
            
        Returns:
            New centroids tensor of shape [n_clusters, n_features]
        """
        # TODO: Implement centroid update
        # For each cluster i:
        #   new_centroid_i = mean(X[labels == i])
        # 
        # Handle empty clusters by keeping old centroid
        pass
    
    def _compute_inertia(self, X: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute sum of squared distances to closest centroid.
        
        Args:
            X: Data tensor of shape [n_samples, n_features]
            labels: Cluster assignments [n_samples]
            
        Returns:
            Inertia (WCSS) value
        """
        # TODO: Compute inertia
        # inertia = Σ ||x - centroid[label[x]]||²
        pass
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Data tensor of shape [n_samples, n_features]
            
        Returns:
            Predicted labels [n_samples]
        """
        # TODO: Assign labels using fitted centroids
        pass
    
    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


class KMeansPlusPlus(KMeans):
    """
    K-Means with K-Means++ initialization.
    
    K-Means++ improves initialization by selecting centroids that are
    far apart from each other. This leads to:
    - Faster convergence
    - Better final clustering
    - More consistent results
    
    Algorithm:
    1. Choose first centroid uniformly at random
    2. For each remaining centroid:
       a. Compute distance D(x) from each point to nearest existing centroid
       b. Choose new centroid with probability proportional to D(x)²
    3. Run standard K-Means with these initial centroids
    
    Args:
        Same as KMeans
    """
    
    def __init__(
        self,
        n_clusters: int = 8,
        max_iters: int = 300,
        tol: float = 1e-4,
        random_seed: Optional[int] = None
    ):
        super().__init__(n_clusters, max_iters, tol, random_seed)
    
    def fit(self, X: torch.Tensor) -> 'KMeansPlusPlus':
        """
        Fit K-Means with K-Means++ initialization.
        
        Args:
            X: Data tensor of shape [n_samples, n_features]
            
        Returns:
            self
        """
        # TODO: Implement K-Means++ initialization
        # 1. Initialize centroids using K-Means++ algorithm:
        #    a. Select first centroid randomly
        #    b. For each remaining centroid:
        #       - Compute squared distance to nearest existing centroid
        #       - Sample new centroid with probability ∝ distance²
        #       Hint: Use torch.multinomial for weighted sampling
        # 
        # 2. Run standard K-Means iteration (same as parent class)
        pass


class MiniBatchKMeans:
    """
    Mini-Batch K-Means for large-scale clustering.
    
    Instead of using all data points in each iteration, use random
    mini-batches for faster updates. This provides:
    - Much faster training on large datasets
    - Lower memory requirements
    - Online learning capability
    - Slight decrease in clustering quality (usually acceptable)
    
    Algorithm:
    1. Initialize centroids (can use K-Means++)
    2. For each mini-batch:
       a. Assign points to nearest centroids
       b. Update centroids using moving average
    3. Repeat for specified number of iterations
    
    Args:
        n_clusters: Number of clusters
        max_iters: Number of mini-batch iterations
        batch_size: Size of mini-batches
        tol: Tolerance for convergence
        init_method: 'random' or 'kmeans++' for initialization
        random_seed: Random seed
    """
    
    def __init__(
        self,
        n_clusters: int = 8,
        max_iters: int = 100,
        batch_size: int = 100,
        tol: float = 1e-4,
        init_method: str = 'kmeans++',
        random_seed: Optional[int] = None
    ):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.tol = tol
        self.init_method = init_method
        self.random_seed = random_seed
        
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self._counts = None  # Count of points assigned to each centroid
    
    def fit(self, X: torch.Tensor) -> 'MiniBatchKMeans':
        """
        Fit Mini-Batch K-Means to data.
        
        Args:
            X: Data tensor of shape [n_samples, n_features]
            
        Returns:
            self
        """
        # TODO: Implement Mini-Batch K-Means
        # 1. Initialize centroids (use _init_centroids method)
        # 
        # 2. Initialize counts for each centroid (for moving average)
        # 
        # 3. For each iteration:
        #    a. Sample random mini-batch
        #    b. Assign mini-batch points to centroids
        #    c. Update centroids using moving average:
        #       count[c] += 1
        #       lr = 1 / count[c]  # learning rate
        #       centroid[c] = (1 - lr) * centroid[c] + lr * mean(mini_batch points assigned to c)
        #    d. Check convergence
        # 
        # 4. After training, assign all points to final clusters
        pass
    
    def _init_centroids(self, X: torch.Tensor) -> torch.Tensor:
        """Initialize centroids using specified method."""
        # TODO: Implement initialization
        # If init_method == 'kmeans++': use K-Means++ initialization
        # Else: randomly sample k points
        pass
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict cluster labels."""
        # TODO: Assign to nearest centroid
        pass
    
    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        """Fit and return labels."""
        self.fit(X)
        return self.labels_


def elbow_method(
    X: torch.Tensor,
    k_range: range,
    n_trials: int = 5
) -> List[Tuple[int, float]]:
    """
    Apply elbow method to find optimal number of clusters.
    
    The elbow method plots inertia vs k and looks for an "elbow"
    where the rate of decrease sharply changes.
    
    Args:
        X: Data tensor [n_samples, n_features]
        k_range: Range of k values to try (e.g., range(2, 11))
        n_trials: Number of random initializations per k
        
    Returns:
        List of (k, best_inertia) tuples
    """
    # TODO: Implement elbow method
    # For each k in k_range:
    #   Run K-Means n_trials times with different seeds
    #   Keep best inertia (lowest)
    # Return list of (k, inertia) pairs
    pass


def silhouette_score(
    X: torch.Tensor,
    labels: torch.Tensor,
    metric: str = 'euclidean'
) -> float:
    """
    Compute silhouette score for clustering quality.
    
    Silhouette score measures how similar a point is to its own cluster
    compared to other clusters. Range: [-1, 1]
    - +1: Point is well-matched to its cluster
    -  0: Point is on border between clusters
    - -1: Point may be assigned to wrong cluster
    
    Formula for point i:
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    where:
    - a(i) = mean distance to points in same cluster
    - b(i) = mean distance to points in nearest other cluster
    
    Args:
        X: Data tensor [n_samples, n_features]
        labels: Cluster assignments [n_samples]
        metric: Distance metric (default: 'euclidean')
        
    Returns:
        Mean silhouette score across all points
    """
    # TODO: Implement silhouette score
    # 1. For each point i:
    #    a. Compute a(i): mean distance to other points in same cluster
    #    b. Compute b(i): mean distance to nearest other cluster
    #    c. Compute s(i) = (b(i) - a(i)) / max(a(i), b(i))
    # 2. Return mean of all s(i)
    pass


# ============= Test Cases =============
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    
    from utils.test_runner import test_ml_implementation
    
    print("=" * 70)
    print("Part 1: Standard K-Means Tests")
    print("=" * 70)
    
    # Generate synthetic data for testing
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create 3 well-separated clusters
    cluster1 = torch.randn(100, 2) * 0.5 + torch.tensor([0.0, 0.0])
    cluster2 = torch.randn(100, 2) * 0.5 + torch.tensor([5.0, 5.0])
    cluster3 = torch.randn(100, 2) * 0.5 + torch.tensor([0.0, 5.0])
    X_test = torch.cat([cluster1, cluster2, cluster3], dim=0)
    
    # K-Means Tests
    def test_kmeans_fit(km):
        km.fit(X_test)
        result = km.centroids is not None and km.labels_ is not None
        expected = True
        return result, expected
    
    def test_kmeans_n_clusters(km):
        km.fit(X_test)
        result = km.centroids.shape[0]
        expected = 3
        return result, expected
    
    def test_kmeans_labels_shape(km):
        km.fit(X_test)
        result = km.labels_.shape[0]
        expected = X_test.shape[0]
        return result, expected
    
    def test_kmeans_convergence(km):
        km.fit(X_test)
        # Should converge in reasonable number of iterations
        result = km.n_iter_ <= km.max_iters
        expected = True
        return result, expected
    
    def test_kmeans_inertia(km):
        km.fit(X_test)
        # Inertia should be positive and finite
        result = km.inertia_ > 0 and torch.isfinite(torch.tensor(km.inertia_))
        expected = True
        return result, expected
    
    def test_kmeans_predict(km):
        km.fit(X_test)
        new_points = torch.tensor([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
        predictions = km.predict(new_points)
        
        # Should assign similar points to same cluster
        result = predictions.shape[0]
        expected = 3
        return result, expected
    
    kmeans = KMeans(n_clusters=3, max_iters=100, random_seed=42)
    
    kmeans_tests = [
        (test_kmeans_fit, "K-Means fits successfully"),
        (test_kmeans_n_clusters, "K-Means creates correct number of clusters"),
        (test_kmeans_labels_shape, "K-Means assigns labels to all points"),
        (test_kmeans_convergence, "K-Means converges"),
        (test_kmeans_inertia, "K-Means computes valid inertia"),
        (test_kmeans_predict, "K-Means predicts new points"),
    ]
    
    test_ml_implementation(kmeans, kmeans_tests)
    
    print("\n" + "=" * 70)
    print("Part 2: K-Means++ Initialization Tests")
    print("=" * 70)
    
    def test_kmeanspp_better_init(kmeanspp):
        # K-Means++ should generally produce better results
        kmeanspp.fit(X_test)
        
        # Compare with random init
        kmeans_random = KMeans(n_clusters=3, max_iters=100, random_seed=42)
        kmeans_random.fit(X_test)
        
        # K-Means++ should converge in fewer iterations (on average)
        result = kmeanspp.inertia_ is not None
        expected = True
        return result, expected
    
    def test_kmeanspp_centroids(kmeanspp):
        kmeanspp.fit(X_test)
        # Centroids should be well-separated initially
        result = kmeanspp.centroids.shape
        expected = (3, 2)
        return result, expected
    
    kmeanspp = KMeansPlusPlus(n_clusters=3, max_iters=100, random_seed=42)
    
    kmeanspp_tests = [
        (test_kmeanspp_better_init, "K-Means++ initialization works"),
        (test_kmeanspp_centroids, "K-Means++ creates correct centroids"),
    ]
    
    test_ml_implementation(kmeanspp, kmeanspp_tests)
    
    print("\n" + "=" * 70)
    print("Part 3: Mini-Batch K-Means Tests")
    print("=" * 70)
    
    # Create larger dataset for mini-batch
    cluster1_large = torch.randn(1000, 2) * 0.5 + torch.tensor([0.0, 0.0])
    cluster2_large = torch.randn(1000, 2) * 0.5 + torch.tensor([5.0, 5.0])
    cluster3_large = torch.randn(1000, 2) * 0.5 + torch.tensor([0.0, 5.0])
    X_large = torch.cat([cluster1_large, cluster2_large, cluster3_large], dim=0)
    
    def test_minibatch_fit(mb_km):
        mb_km.fit(X_large)
        result = mb_km.centroids is not None
        expected = True
        return result, expected
    
    def test_minibatch_faster(mb_km):
        # Mini-batch should handle large data
        mb_km.fit(X_large)
        result = mb_km.labels_.shape[0]
        expected = X_large.shape[0]
        return result, expected
    
    def test_minibatch_predict(mb_km):
        mb_km.fit(X_large)
        predictions = mb_km.predict(X_test)
        result = predictions.shape[0]
        expected = X_test.shape[0]
        return result, expected
    
    minibatch_km = MiniBatchKMeans(
        n_clusters=3,
        max_iters=50,
        batch_size=100,
        random_seed=42
    )
    
    minibatch_tests = [
        (test_minibatch_fit, "Mini-Batch K-Means fits"),
        (test_minibatch_faster, "Mini-Batch K-Means handles large data"),
        (test_minibatch_predict, "Mini-Batch K-Means predicts"),
    ]
    
    test_ml_implementation(minibatch_km, minibatch_tests)
    
    print("\n" + "=" * 70)
    print("Implementation Tips:")
    print("=" * 70)
    print("""
K-Means Algorithm:
1. Initialize: centroids = X[random_indices]
2. Repeat:
   - Assign: labels = argmin_c ||x - centroid_c||²
   - Update: centroid_c = mean(X[labels == c])
3. Stop when: ||new_centroids - old_centroids|| < tol

K-Means++ Initialization:
1. centroid_0 = random point from X
2. For i = 1 to k-1:
   - D² = squared distance to nearest existing centroid
   - P(x) = D²(x) / sum(D²)  # probability for each point
   - centroid_i = sample from X with probability P

Mini-Batch K-Means Update:
- counts[c] = number of points ever assigned to cluster c
- learning_rate = 1 / counts[c]
- centroid[c] = (1 - lr) * centroid[c] + lr * mean(batch_points[c])

Distance Computation (efficient):
- ||x - c||² = ||x||² + ||c||² - 2(x·c)
- Or use: torch.cdist(X, centroids, p=2)

Silhouette Score:
- a(i) = mean distance to same cluster
- b(i) = min mean distance to other clusters
- s(i) = (b - a) / max(a, b)
    """)
    
    print("\n" + "=" * 70)
    print("Practical Applications:")
    print("=" * 70)
    print("""
K-Means Use Cases:

1. Customer Segmentation:
   - Cluster customers by behavior
   - Target marketing campaigns
   
2. Image Compression:
   - Reduce colors to k representative values
   - Vector quantization
   
3. Document Clustering:
   - Group similar documents
   - Topic discovery
   
4. Anomaly Detection:
   - Points far from all centroids are anomalies
   
5. Feature Engineering:
   - Cluster distances as features
   - For downstream ML models

When to Use Each Variant:
- Standard K-Means: Small to medium datasets (< 10K samples)
- K-Means++: Always use this initialization!
- Mini-Batch K-Means: Large datasets (> 100K samples)
    """)

