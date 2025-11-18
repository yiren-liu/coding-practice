"""
ML Implementation: K-Means Clustering

Description:
Implement the K-Means clustering algorithm from scratch using NumPy. K-Means is an 
unsupervised learning algorithm that partitions data into K distinct clusters by 
iteratively assigning points to the nearest centroid and updating centroids.

Algorithm Steps:
1. Initialize K centroids (randomly or using K-Means++)
2. Assign each point to the nearest centroid
3. Update centroids as the mean of assigned points
4. Repeat steps 2-3 until convergence

Key Requirements:
- Support for multiple initialization methods (random, k-means++)
- Euclidean distance computation
- Convergence detection
- Handle edge cases (empty clusters)
- Efficient vectorized operations

Applications:
- Customer segmentation
- Image compression
- Data preprocessing
- Anomaly detection

References:
- Original paper: MacQueen (1967)
- K-Means++: Arthur & Vassilvitskii (2007) https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
"""

import numpy as np
from typing import Optional, Tuple, Literal
import warnings


class KMeans:
    """
    K-Means clustering implementation.
    
    Args:
        n_clusters: Number of clusters (K)
        max_iters: Maximum number of iterations (default: 300)
        tol: Tolerance for convergence (default: 1e-4)
        init: Initialization method - 'random' or 'kmeans++' (default: 'kmeans++')
        n_init: Number of times to run with different initializations (default: 10)
        random_state: Random seed for reproducibility (default: None)
    """
    
    def __init__(
        self,
        n_clusters: int,
        max_iters: int = 300,
        tol: float = 1e-4,
        init: Literal['random', 'kmeans++'] = 'kmeans++',
        n_init: int = 10,
        random_state: Optional[int] = None
    ):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        
        # These will be set after fitting
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None  # Sum of squared distances to closest centroid
        self.n_iter_ = None   # Number of iterations run
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        Fit K-Means clustering on the data.
        
        Args:
            X: Data matrix of shape [n_samples, n_features]
            
        Returns:
            self: Fitted KMeans instance
        """
        n_samples, n_features = X.shape
        
        if self.n_clusters > n_samples:
            raise ValueError(f"n_clusters ({self.n_clusters}) cannot be larger than n_samples ({n_samples})")
        
        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Run K-Means multiple times and keep the best result
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_n_iter = None
        
        for _ in range(self.n_init):
            # Initialize centroids
            init_centroids = self._initialize_centroids(X)
            
            # Run single K-Means
            centroids, labels, inertia, n_iter = self._fit_single(X, init_centroids)
            
            # Keep the best result
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = n_iter
        
        # Store the best results
        self.centroids = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        
        return self
    
    def _fit_single(self, X: np.ndarray, init_centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Run a single K-Means clustering with given initial centroids.
        
        Args:
            X: Data matrix of shape [n_samples, n_features]
            init_centroids: Initial centroids of shape [n_clusters, n_features]
            
        Returns:
            centroids: Final centroids [n_clusters, n_features]
            labels: Cluster labels [n_samples]
            inertia: Sum of squared distances
            n_iter: Number of iterations performed
        """
        centroids = init_centroids.copy()
        
        for iteration in range(self.max_iters):
            # Assign points to nearest centroid
            labels = self._assign_clusters(X, centroids)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check for convergence
            centroid_shift = np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max()
            
            centroids = new_centroids
            
            if centroid_shift < self.tol:
                break
        
        # Compute final inertia
        inertia = self._compute_inertia(X, labels, centroids)
        
        return centroids, labels, inertia, iteration + 1
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using the specified method.
        
        Args:
            X: Data matrix of shape [n_samples, n_features]
            
        Returns:
            centroids: Initial centroids of shape [n_clusters, n_features]
        """
        n_samples, n_features = X.shape
        
        if self.init == 'random':
            # Randomly select K points from X
            indices = np.random.choice(n_samples, size=self.n_clusters, replace=False)
            centroids = X[indices].copy()
            
        elif self.init == 'kmeans++':
            # K-Means++ initialization
            centroids = np.zeros((self.n_clusters, n_features))
            
            # Choose first centroid uniformly at random
            centroids[0] = X[np.random.randint(n_samples)]
            
            # Choose remaining centroids
            for k in range(1, self.n_clusters):
                # Compute distances to nearest existing centroid
                distances = self._compute_distances(X, centroids[:k])
                min_distances = distances.min(axis=1)
                
                # Square the distances for probability computation
                squared_distances = min_distances ** 2
                
                # Choose next centroid with probability proportional to squared distance
                probabilities = squared_distances / squared_distances.sum()
                next_centroid_idx = np.random.choice(n_samples, p=probabilities)
                centroids[k] = X[next_centroid_idx]
            
        else:
            raise ValueError(f"Unknown init method: {self.init}")
        
        return centroids
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each point to the nearest centroid.
        
        Args:
            X: Data matrix of shape [n_samples, n_features]
            centroids: Centroids of shape [n_clusters, n_features]
            
        Returns:
            labels: Cluster assignments of shape [n_samples]
        """
        # Compute distances to all centroids
        distances = self._compute_distances(X, centroids)
        
        # Assign to nearest centroid
        labels = np.argmin(distances, axis=1)
        
        return labels
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update centroids as the mean of assigned points.
        
        Args:
            X: Data matrix of shape [n_samples, n_features]
            labels: Current cluster labels of shape [n_samples]
            
        Returns:
            new_centroids: Updated centroids of shape [n_clusters, n_features]
        """
        n_features = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features))
        
        for k in range(self.n_clusters):
            # Find all points assigned to cluster k
            cluster_points = X[labels == k]
            
            if len(cluster_points) > 0:
                # Update centroid as mean of assigned points
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                # Handle empty cluster: reinitialize randomly
                warnings.warn(f"Cluster {k} is empty. Reinitializing randomly.")
                new_centroids[k] = X[np.random.randint(len(X))]
        
        return new_centroids
    
    def _compute_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distances between all points and centroids.
        
        Args:
            X: Data matrix of shape [n_samples, n_features]
            centroids: Centroids of shape [n_clusters, n_features]
            
        Returns:
            distances: Distance matrix of shape [n_samples, n_clusters]
                      distances[i, j] = distance from point i to centroid j
        """
        # Efficient computation using broadcasting
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        
        # X: [n_samples, n_features]
        # centroids: [n_clusters, n_features]
        
        # Method 1: Using broadcasting directly (clear but may use more memory)
        # distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        
        # Method 2: Using the expanded form (more memory efficient)
        X_sq = (X ** 2).sum(axis=1, keepdims=True)  # [n_samples, 1]
        C_sq = (centroids ** 2).sum(axis=1, keepdims=True).T  # [1, n_clusters]
        XC = X @ centroids.T  # [n_samples, n_clusters]
        
        distances = np.sqrt(np.maximum(X_sq + C_sq - 2 * XC, 0))  # Avoid negative due to numerical errors
        
        return distances
    
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """
        Compute inertia (sum of squared distances to nearest centroid).
        
        Args:
            X: Data matrix of shape [n_samples, n_features]
            labels: Cluster labels of shape [n_samples]
            centroids: Centroids of shape [n_clusters, n_features]
            
        Returns:
            inertia: Sum of squared distances
        """
        # Get the centroid for each point
        assigned_centroids = centroids[labels]
        
        # Compute squared distances
        squared_distances = ((X - assigned_centroids) ** 2).sum(axis=1)
        
        # Sum to get inertia
        inertia = squared_distances.sum()
        
        return inertia
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Data matrix of shape [n_samples, n_features]
            
        Returns:
            labels: Predicted cluster labels of shape [n_samples]
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Assign to nearest centroid
        labels = self._assign_clusters(X, self.centroids)
        
        return labels
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and return cluster labels.
        
        Args:
            X: Data matrix of shape [n_samples, n_features]
            
        Returns:
            labels: Cluster labels of shape [n_samples]
        """
        self.fit(X)
        return self.labels_
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X to cluster-distance space.
        
        Args:
            X: Data matrix of shape [n_samples, n_features]
            
        Returns:
            distances: Distances to each centroid of shape [n_samples, n_clusters]
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Return distances to all centroids
        distances = self._compute_distances(X, self.centroids)
        
        return distances


# ============= Test Cases =============
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    
    from utils.test_runner import test_ml_implementation
    
    # Generate test data
    np.random.seed(42)
    
    # Create 3 well-separated clusters
    cluster1 = np.random.randn(100, 2) + np.array([0, 0])
    cluster2 = np.random.randn(100, 2) + np.array([5, 5])
    cluster3 = np.random.randn(100, 2) + np.array([10, 0])
    test_data = np.vstack([cluster1, cluster2, cluster3])
    
    # Test 1: Output shape
    def test_output_shape(kmeans):
        X = test_data
        labels = kmeans.fit_predict(X)
        expected_shape = (300,)
        return labels.shape, expected_shape
    
    # Test 2: Number of clusters
    def test_num_clusters(kmeans):
        X = test_data
        kmeans.fit(X)
        n_unique_labels = len(np.unique(kmeans.labels_))
        expected = 3
        return n_unique_labels, expected
    
    # Test 3: Centroids shape
    def test_centroids_shape(kmeans):
        X = test_data
        kmeans.fit(X)
        result = kmeans.centroids.shape
        expected = (3, 2)  # 3 clusters, 2 features
        return result, expected
    
    # Test 4: Predict on new data
    def test_predict(kmeans):
        X = test_data
        kmeans.fit(X)
        
        # Create new points close to existing clusters
        new_points = np.array([[0, 0], [5, 5], [10, 0]])
        predictions = kmeans.predict(new_points)
        
        # All predictions should be valid cluster IDs
        result = np.all((predictions >= 0) & (predictions < 3))
        expected = True
        return result, expected
    
    # Test 5: Inertia decreases or stays same
    def test_inertia_convergence(kmeans):
        X = test_data
        
        # Track inertia through iterations by running manual fit
        kmeans.fit(X)
        
        # Inertia should be computed
        result = kmeans.inertia_ is not None and kmeans.inertia_ >= 0
        expected = True
        return result, expected
    
    # Test 6: Labels are integers in valid range
    def test_valid_labels(kmeans):
        X = test_data
        labels = kmeans.fit_predict(X)
        
        result = (labels.dtype == np.int64 or labels.dtype == np.int32) and \
                 np.all(labels >= 0) and np.all(labels < 3)
        expected = True
        return result, expected
    
    # Test 7: Transform returns distances
    def test_transform(kmeans):
        X = test_data[:10]  # Use subset for speed
        kmeans.fit(test_data)
        
        distances = kmeans.transform(X)
        result = distances.shape
        expected = (10, 3)  # 10 samples, 3 clusters
        return result, expected
    
    # Test 8: Deterministic with random_state
    def test_reproducibility(kmeans):
        X = test_data
        
        kmeans1 = KMeans(n_clusters=3, random_state=42, n_init=1)
        kmeans2 = KMeans(n_clusters=3, random_state=42, n_init=1)
        
        labels1 = kmeans1.fit_predict(X)
        labels2 = kmeans2.fit_predict(X)
        
        result = np.array_equal(labels1, labels2)
        expected = True
        return result, expected
    
    # Test 9: K-Means++ initialization
    def test_kmeans_plus_plus(kmeans):
        X = test_data
        kmeans_pp = KMeans(n_clusters=3, init='kmeans++', random_state=42, n_init=1)
        kmeans_pp.fit(X)
        
        # Should converge and have valid centroids
        result = kmeans_pp.centroids is not None and kmeans_pp.centroids.shape == (3, 2)
        expected = True
        return result, expected
    
    # Test 10: Convergence (n_iter should be set)
    def test_convergence_tracking(kmeans):
        X = test_data
        kmeans.fit(X)
        
        result = kmeans.n_iter_ is not None and kmeans.n_iter_ > 0
        expected = True
        return result, expected
    
    print("=" * 60)
    print("K-Means Clustering Tests")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  n_clusters: 3")
    print(f"  data_points: 300 (3 clusters of 100 points)")
    print(f"  features: 2D")
    print()
    
    # Create instance
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=5)
    
    tests = [
        (test_output_shape, "Output shape test"),
        (test_num_clusters, "Number of clusters"),
        (test_centroids_shape, "Centroids shape"),
        (test_predict, "Predict on new data"),
        (test_inertia_convergence, "Inertia computation"),
        (test_valid_labels, "Valid label range"),
        (test_transform, "Transform to distances"),
        (test_reproducibility, "Reproducibility with random_state"),
        (test_kmeans_plus_plus, "K-Means++ initialization"),
        (test_convergence_tracking, "Convergence tracking"),
    ]
    
    test_ml_implementation(kmeans, tests)
    
    print("\n" + "=" * 60)
    print("Implementation Tips:")
    print("=" * 60)
    print("""
1. Distance Computation (Efficient):
   - Use broadcasting: distances = np.sqrt(((X[:, None] - centroids) ** 2).sum(axis=2))
   - Or use the formula: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
   
2. K-Means++ Initialization:
   - First centroid: random point from X
   - Next centroids: Choose with probability ∝ distance^2 to nearest centroid
   - Use np.random.choice with probability weights
   
3. Convergence:
   - Check if centroids change by less than tol
   - Or check if labels don't change between iterations
   - Use np.allclose() for numerical comparison
   
4. Empty Clusters:
   - If a cluster has no points, reinitialize its centroid
   - Options: random point, farthest point, or keep old centroid
   
5. Vectorization:
   - Avoid loops over samples when possible
   - Use NumPy broadcasting and vectorized operations
   - This is crucial for performance on large datasets
    """)

