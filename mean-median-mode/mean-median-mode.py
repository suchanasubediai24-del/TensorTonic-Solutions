
from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """

    # Convert list into NumPy array
    arr = np.asarray(x, dtype=float)

    # Calculate mean
    mean = float(np.mean(arr))

    # Calculate median
    median = float(np.median(arr))

    # Calculate mode
    counts = Counter(arr)

    max_count = max(counts.values())

    # Select smallest value in case of a tie
    mode = float(min(
        key for key, value in counts.items()
        if value == max_count
    ))

    # Return results
    return {
        "mean": mean,
        "median": median,
        "mode": mode
    }


# Test Case 1
print(mean_median_mode([1, 2, 3, 4, 5]))

# Test Case 2
print(mean_median_mode([1, 2, 2, 3, 4]))

# Test Case 3
print(mean_median_mode([1, 2, 3, 4]))
