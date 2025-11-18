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
        # TODO: Implement the fit method
        # 1. Run K-Means n_init times with different initializations
        # 2. Keep the result with the lowest inertia
        # 3. Store the best centroids, labels, and inertia
        
        # Hint: Use _fit_single() to run one K-Means iteration
        # and keep track of the best result
        
        pass
    
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
        # TODO: Implement single K-Means run
        # 1. Start with init_centroids
        # 2. For each iteration:
        #    a. Assign points to nearest centroid
        #    b. Update centroids
        #    c. Check for convergence
        # 3. Return final centroids, labels, inertia, and iteration count
        
        pass
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using the specified method.
        
        Args:
            X: Data matrix of shape [n_samples, n_features]
            
        Returns:
            centroids: Initial centroids of shape [n_clusters, n_features]
        """
        # TODO: Implement initialization
        # Support two methods:
        # 1. 'random': Randomly select K points from X
        # 2. 'kmeans++': Use K-Means++ initialization
        
        if self.init == 'random':
            # TODO: Random initialization
            pass
        elif self.init == 'kmeans++':
            # TODO: K-Means++ initialization
            # Algorithm:
            # 1. Choose first centroid uniformly at random from X
            # 2. For each remaining centroid:
            #    a. Compute distance of each point to nearest existing centroid
            #    b. Choose next centroid with probability proportional to squared distance
            pass
        else:
            raise ValueError(f"Unknown init method: {self.init}")
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each point to the nearest centroid.
        
        Args:
            X: Data matrix of shape [n_samples, n_features]
            centroids: Centroids of shape [n_clusters, n_features]
            
        Returns:
            labels: Cluster assignments of shape [n_samples]
        """
        # TODO: Assign each point to nearest centroid
        # Hint: Use _compute_distances() and np.argmin()
        pass
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update centroids as the mean of assigned points.
        
        Args:
            X: Data matrix of shape [n_samples, n_features]
            labels: Current cluster labels of shape [n_samples]
            
        Returns:
            new_centroids: Updated centroids of shape [n_clusters, n_features]
        """
        # TODO: Update centroids
        # For each cluster, compute the mean of all assigned points
        # Handle empty clusters by keeping the old centroid or reinitializing
        
        pass
    
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
        # TODO: Compute distances efficiently using broadcasting
        # Hint: Use the formula: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        # This avoids explicit loops and is much faster
        
        pass
    
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
        # TODO: Compute inertia
        # Sum of squared distances from each point to its assigned centroid
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Data matrix of shape [n_samples, n_features]
            
        Returns:
            labels: Predicted cluster labels of shape [n_samples]
        """
        # TODO: Predict cluster assignments for new data
        # Assign each point to the nearest centroid
        
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        pass
    
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
        # TODO: Return distances to all centroids
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        pass


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

