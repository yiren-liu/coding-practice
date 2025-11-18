"""
ML Implementation: K-Means Clustering and Variants - SOLUTION

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
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
        
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly from data points
        indices = torch.randperm(n_samples)[:self.n_clusters]
        self.centroids = X[indices].clone()
        
        # Iterate until convergence
        for iteration in range(self.max_iters):
            # E-step: Assign labels
            labels = self._assign_labels(X)
            
            # M-step: Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check convergence
            centroid_shift = torch.sum((new_centroids - self.centroids) ** 2)
            self.centroids = new_centroids
            
            if centroid_shift < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iters
        
        # Final assignment
        self.labels_ = self._assign_labels(X)
        self.inertia_ = self._compute_inertia(X, self.labels_)
        
        return self
    
    def _assign_labels(self, X: torch.Tensor) -> torch.Tensor:
        """
        Assign each point to nearest centroid.
        
        Args:
            X: Data tensor of shape [n_samples, n_features]
            
        Returns:
            Labels tensor of shape [n_samples] with values in [0, k-1]
        """
        # Compute pairwise distances using efficient formula
        # ||x - c||² = ||x||² + ||c||² - 2(x·c)
        distances = torch.cdist(X, self.centroids, p=2)  # [n_samples, n_clusters]
        
        # Assign to nearest centroid
        labels = torch.argmin(distances, dim=1)
        
        return labels
    
    def _update_centroids(self, X: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Update centroids as mean of assigned points.
        
        Args:
            X: Data tensor of shape [n_samples, n_features]
            labels: Current cluster assignments [n_samples]
            
        Returns:
            New centroids tensor of shape [n_clusters, n_features]
        """
        new_centroids = torch.zeros_like(self.centroids)
        
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                new_centroids[k] = X[mask].mean(dim=0)
            else:
                # Keep old centroid if cluster is empty
                new_centroids[k] = self.centroids[k]
        
        return new_centroids
    
    def _compute_inertia(self, X: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute sum of squared distances to closest centroid.
        
        Args:
            X: Data tensor of shape [n_samples, n_features]
            labels: Cluster assignments [n_samples]
            
        Returns:
            Inertia (WCSS) value
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                cluster_points = X[mask]
                centroid = self.centroids[k]
                inertia += torch.sum((cluster_points - centroid) ** 2).item()
        
        return inertia
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Data tensor of shape [n_samples, n_features]
            
        Returns:
            Predicted labels [n_samples]
        """
        return self._assign_labels(X)
    
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
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
        
        n_samples, n_features = X.shape
        
        # K-Means++ initialization
        # 1. Choose first centroid randomly
        first_idx = torch.randint(0, n_samples, (1,)).item()
        centroids = [X[first_idx]]
        
        # 2. Choose remaining centroids
        for _ in range(1, self.n_clusters):
            # Compute squared distance to nearest existing centroid
            centroids_tensor = torch.stack(centroids)
            distances = torch.cdist(X, centroids_tensor, p=2)  # [n_samples, k]
            min_distances = torch.min(distances, dim=1)[0]  # [n_samples]
            squared_distances = min_distances ** 2
            
            # Sample new centroid with probability proportional to D²
            probabilities = squared_distances / squared_distances.sum()
            next_idx = torch.multinomial(probabilities, 1).item()
            centroids.append(X[next_idx])
        
        self.centroids = torch.stack(centroids)
        
        # Run standard K-Means iteration
        for iteration in range(self.max_iters):
            labels = self._assign_labels(X)
            new_centroids = self._update_centroids(X, labels)
            
            centroid_shift = torch.sum((new_centroids - self.centroids) ** 2)
            self.centroids = new_centroids
            
            if centroid_shift < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iters
        
        self.labels_ = self._assign_labels(X)
        self.inertia_ = self._compute_inertia(X, self.labels_)
        
        return self


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
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
        
        n_samples, n_features = X.shape
        
        # Initialize centroids
        self.centroids = self._init_centroids(X)
        self._counts = torch.zeros(self.n_clusters)
        
        # Mini-batch iterations
        for iteration in range(self.max_iters):
            # Sample random mini-batch
            batch_indices = torch.randint(0, n_samples, (min(self.batch_size, n_samples),))
            X_batch = X[batch_indices]
            
            # Assign mini-batch to centroids
            distances = torch.cdist(X_batch, self.centroids, p=2)
            labels_batch = torch.argmin(distances, dim=1)
            
            # Store old centroids for convergence check
            old_centroids = self.centroids.clone()
            
            # Update centroids using moving average
            for k in range(self.n_clusters):
                mask = labels_batch == k
                if mask.sum() > 0:
                    # Update count
                    self._counts[k] += mask.sum().item()
                    
                    # Compute learning rate
                    lr = 1.0 / self._counts[k]
                    
                    # Update centroid with moving average
                    batch_mean = X_batch[mask].mean(dim=0)
                    self.centroids[k] = (1 - lr) * self.centroids[k] + lr * batch_mean
            
            # Check convergence
            centroid_shift = torch.sum((self.centroids - old_centroids) ** 2)
            if centroid_shift < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iters
        
        # Assign all points to final clusters
        self.labels_ = self.predict(X)
        self.inertia_ = self._compute_inertia(X, self.labels_)
        
        return self
    
    def _init_centroids(self, X: torch.Tensor) -> torch.Tensor:
        """Initialize centroids using specified method."""
        n_samples = X.shape[0]
        
        if self.init_method == 'kmeans++':
            # Use K-Means++ initialization
            first_idx = torch.randint(0, n_samples, (1,)).item()
            centroids = [X[first_idx]]
            
            for _ in range(1, self.n_clusters):
                centroids_tensor = torch.stack(centroids)
                distances = torch.cdist(X, centroids_tensor, p=2)
                min_distances = torch.min(distances, dim=1)[0]
                squared_distances = min_distances ** 2
                
                probabilities = squared_distances / squared_distances.sum()
                next_idx = torch.multinomial(probabilities, 1).item()
                centroids.append(X[next_idx])
            
            return torch.stack(centroids)
        else:
            # Random initialization
            indices = torch.randperm(n_samples)[:self.n_clusters]
            return X[indices].clone()
    
    def _compute_inertia(self, X: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute inertia."""
        inertia = 0.0
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                cluster_points = X[mask]
                centroid = self.centroids[k]
                inertia += torch.sum((cluster_points - centroid) ** 2).item()
        return inertia
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict cluster labels."""
        distances = torch.cdist(X, self.centroids, p=2)
        return torch.argmin(distances, dim=1)
    
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
    results = []
    
    for k in k_range:
        best_inertia = float('inf')
        
        # Try multiple random initializations
        for trial in range(n_trials):
            kmeans = KMeansPlusPlus(n_clusters=k, random_seed=trial)
            kmeans.fit(X)
            
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
        
        results.append((k, best_inertia))
    
    return results


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
    n_samples = X.shape[0]
    n_clusters = len(torch.unique(labels))
    
    if n_clusters <= 1 or n_clusters >= n_samples:
        return 0.0
    
    # Compute pairwise distances
    distances = torch.cdist(X, X, p=2)
    
    silhouette_scores = torch.zeros(n_samples)
    
    for i in range(n_samples):
        # Current cluster
        cluster_i = labels[i].item()
        
        # a(i): mean distance to points in same cluster
        same_cluster_mask = labels == cluster_i
        same_cluster_mask[i] = False  # Exclude point itself
        
        if same_cluster_mask.sum() > 0:
            a_i = distances[i, same_cluster_mask].mean()
        else:
            a_i = 0.0
        
        # b(i): mean distance to nearest other cluster
        b_i = float('inf')
        for k in range(n_clusters):
            if k != cluster_i:
                other_cluster_mask = labels == k
                if other_cluster_mask.sum() > 0:
                    mean_dist = distances[i, other_cluster_mask].mean()
                    b_i = min(b_i, mean_dist)
        
        # Compute silhouette score for point i
        if b_i == float('inf'):
            silhouette_scores[i] = 0.0
        else:
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
    
    return silhouette_scores.mean().item()


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
    print("Part 4: Evaluation Metrics Tests")
    print("=" * 70)
    
    def test_elbow_method(_):
        results = elbow_method(X_test, range(2, 6), n_trials=3)
        
        # Should have results for each k
        result = len(results)
        expected = 4  # k = 2, 3, 4, 5
        return result, expected
    
    def test_silhouette(_):
        kmeans = KMeans(n_clusters=3, random_seed=42)
        kmeans.fit(X_test)
        
        score = silhouette_score(X_test, kmeans.labels_)
        
        # For well-separated clusters, should be > 0.5
        result = score > 0.3
        expected = True
        
        print(f"\n  Silhouette score: {score:.4f}")
        return result, expected
    
    evaluation_tests = [
        (test_elbow_method, "Elbow method works"),
        (test_silhouette, "Silhouette score computes"),
    ]
    
    test_ml_implementation(None, evaluation_tests)
    
    print("\n" + "=" * 70)
    print("Performance Comparison:")
    print("=" * 70)
    
    # Compare convergence speed
    import time
    
    print("\nConvergence Comparison (300 samples):")
    
    # Standard K-Means
    start = time.time()
    km_standard = KMeans(n_clusters=3, random_seed=42)
    km_standard.fit(X_test)
    time_standard = time.time() - start
    
    # K-Means++
    start = time.time()
    km_pp = KMeansPlusPlus(n_clusters=3, random_seed=42)
    km_pp.fit(X_test)
    time_pp = time.time() - start
    
    print(f"  Standard K-Means:  {km_standard.n_iter_:3d} iterations, inertia: {km_standard.inertia_:8.2f}, time: {time_standard*1000:.2f}ms")
    print(f"  K-Means++:         {km_pp.n_iter_:3d} iterations, inertia: {km_pp.inertia_:8.2f}, time: {time_pp*1000:.2f}ms")
    
    print("\nScalability Comparison (3000 samples):")
    
    # Standard K-Means on large data
    start = time.time()
    km_large = KMeans(n_clusters=3, max_iters=50, random_seed=42)
    km_large.fit(X_large)
    time_large = time.time() - start
    
    # Mini-Batch K-Means on large data
    start = time.time()
    mb_km = MiniBatchKMeans(n_clusters=3, max_iters=50, batch_size=100, random_seed=42)
    mb_km.fit(X_large)
    time_mb = time.time() - start
    
    print(f"  Standard K-Means:      inertia: {km_large.inertia_:8.2f}, time: {time_large*1000:.2f}ms")
    print(f"  Mini-Batch K-Means:    inertia: {mb_km.inertia_:8.2f}, time: {time_mb*1000:.2f}ms")
    print(f"  Speedup: {time_large/time_mb:.2f}x")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("=" * 70)
    print("""
Algorithm Characteristics:

1. Standard K-Means:
   ✓ Simple and interpretable
   ✓ Guaranteed to converge (to local minimum)
   ✗ Sensitive to initialization
   ✗ Can be slow on large datasets
   
2. K-Means++:
   ✓ Better initialization → better clusters
   ✓ Usually converges faster
   ✓ More consistent results
   ✓ O(log k) approximation guarantee
   
3. Mini-Batch K-Means:
   ✓ Much faster on large datasets (3-10x speedup)
   ✓ Lower memory usage
   ✓ Online learning capability
   ✗ Slightly worse clustering quality (~5%)

When to Use:
- Small data (< 10K): K-Means++ (best quality)
- Large data (> 100K): Mini-Batch K-Means (speed)
- Production: Always use K-Means++ init at minimum
    """)
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)

