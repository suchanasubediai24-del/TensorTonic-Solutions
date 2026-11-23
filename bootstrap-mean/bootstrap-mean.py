
import numpy as np

def bootstrap_mean(x: list, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """
    Returns a dictionary with bootstrap_mean, lower, and upper.
    """

    # Convert input into NumPy array
    x = np.asarray(x, dtype=float)

    # Initialize random generator
    rng = np.random.default_rng(seed)

    # Get dataset size
    n = len(x)

    # Generate bootstrap samples with replacement
    indices = rng.integers(0, n, size=(n_bootstrap, n))

    # Calculate mean of each bootstrap sample
    bootstrap_samples = x[indices]
    bootstrap_means = np.mean(bootstrap_samples, axis=1)

    # Calculate average bootstrap mean
    mean = float(np.mean(bootstrap_means))

    # Calculate confidence interval
    alpha = (1 - ci) / 2

    lower = float(np.quantile(bootstrap_means, alpha))
    upper = float(np.quantile(bootstrap_means, 1 - alpha))

    # Return results
    return {
        "bootstrap_mean": mean,
        "lower": lower,
        "upper": upper
    }
