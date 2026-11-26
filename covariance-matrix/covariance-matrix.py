
import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    # Convert input into a NumPy array
    X = np.asarray(X, dtype=float)

    # Calculate mean of each feature
    mean = np.mean(X, axis=0)

    # Center the data
    X_centered = X - mean

    # Calculate sample covariance matrix
    cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)

    return cov
