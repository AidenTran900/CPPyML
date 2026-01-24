"""
Example script to test the Python bindings for the ML library.
"""

import numpy as np
import ml_lib


def test_matrix():
    """Test Matrix creation and conversion."""
    print("Testing Matrix...")

    # Create matrix from numpy array
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    mat = ml_lib.Matrix(arr)

    assert mat.rows() == 2
    assert mat.cols() == 2

    # Convert back to numpy
    result = mat.to_numpy()
    assert np.allclose(arr, result)

    print("  Matrix tests passed!")


def test_kmeans():
    """Test K-Means clustering."""
    print("Testing KMeansClustering...")

    # Create simple clustered data
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(10, 2) + [0, 0],
        np.random.randn(10, 2) + [5, 5],
    ])

    kmeans = ml_lib.KMeansClustering(k=2, max_iter=100)
    kmeans.fit(X, np.zeros((X.shape[0], 1)))  # y is unused for clustering

    predictions = kmeans.predict(X)
    assert predictions.shape == (20, 1)

    print("  KMeansClustering tests passed!")


def test_knn():
    """Test K-Nearest Neighbors."""
    print("Testing KNearestNeighbors...")

    # Simple classification data
    X_train = np.array([[0, 0], [1, 1], [2, 2], [10, 10], [11, 11], [12, 12]], dtype=np.float64)
    y_train = np.array([[0], [0], [0], [1], [1], [1]], dtype=np.float64)

    knn = ml_lib.KNearestNeighbors(k=3)
    knn.fit(X_train, y_train)

    X_test = np.array([[0.5, 0.5], [10.5, 10.5]], dtype=np.float64)
    predictions = knn.predict(X_test)

    assert predictions.shape == (2, 1)

    print("  KNearestNeighbors tests passed!")


def test_metrics():
    """Test evaluation metrics."""
    print("Testing metrics...")

    y_true = np.array([1, 0, 1, 1, 0, 1], dtype=np.float64)
    y_pred = np.array([1, 0, 1, 0, 0, 1], dtype=np.float64)

    accuracy = ml_lib.metrics.accuracy(y_true, y_pred)
    assert 0 <= accuracy <= 1

    mse = ml_lib.metrics.mse(y_true, y_pred)
    assert mse >= 0

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  MSE: {mse:.4f}")
    print("  Metrics tests passed!")


def main():
    print("=" * 50)
    print("ML Library Python Bindings Test")
    print("=" * 50)

    test_matrix()
    test_kmeans()
    test_knn()
    test_metrics()

    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
