
import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """

    # Convert list into NumPy array
    arr = np.asarray(x, dtype=float)

    # Calculate sample variance (ddof=1)
    variance = float(np.var(arr, ddof=1))

    # Calculate sample standard deviation
    standard_deviation = float(np.sqrt(variance))

    # Return results
    return {
        "variance": variance,
        "standard_deviation": standard_deviation
    }


# Test Case 1
print(sample_var_std([1, 2, 3]))

# Test Case 2
print(sample_var_std([5, 7]))

# Test Case 3
print(sample_var_std([4, 4, 4, 4]))
