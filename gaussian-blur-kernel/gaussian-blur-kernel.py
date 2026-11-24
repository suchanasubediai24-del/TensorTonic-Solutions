
import math

def gaussian_kernel(size: int, sigma: float) -> list:
    """
    Returns a square two-dimensional list.
    """
    kernel = []
    center = size // 2

    for i in range(size):
        row = []

        for j in range(size):
            x = i - center
            y = j - center

            value = math.exp(-(x*x + y*y) / (2 * sigma*sigma))
            row.append(value)

        kernel.append(row)

    total = sum(sum(row) for row in kernel)

    for i in range(size):
        for j in range(size):
            kernel[i][j] /= total

    return kernel
