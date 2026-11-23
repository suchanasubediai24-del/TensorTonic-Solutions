
import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """

    # Convert lists into NumPy arrays
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # Calculate dot product
    dot_product = np.dot(a, b)

    # Calculate magnitudes
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    # Handle zero vectors
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    # Calculate cosine similarity
    similarity = dot_product / (magnitude_a * magnitude_b)

    return float(similarity)


# Example
a = [1, 2, 3]
b = [4, 5, 6]

result = cosine_similarity(a, b)

print("Cosine Similarity:", result)
