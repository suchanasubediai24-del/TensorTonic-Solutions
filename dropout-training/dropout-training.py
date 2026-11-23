
import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:

    # Convert input to NumPy array
    x = np.asarray(x, dtype=float)

    # Validate dropout probability
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")

    # Initialize random generator
    if rng is None:
        rng = np.random.default_rng()

    # Generate dropout mask
    mask = (rng.random(x.shape) >= p).astype(float)

    # Apply inverted dropout
    if p == 1:
        dropout_pattern = np.zeros_like(x)
    else:
        dropout_pattern = mask / (1 - p)

    # Calculate output
    output = x * dropout_pattern

    return output, dropout_pattern


# Test Case 1
x = [[1, 2], [3, 4]]
rng = np.random.default_rng(123)

output, pattern = dropout(x, p=0.5, rng=rng)

print("Output:", output.tolist())
print("Dropout Pattern:", pattern.tolist())


# Test Case 2
x = [[1, 2], [3, 4]]
rng = np.random.default_rng(7)

output, pattern = dropout(x, p=0.0, rng=rng)

print("Output:", output.tolist())
print("Dropout Pattern:", pattern.tolist())
