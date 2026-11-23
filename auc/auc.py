
import numpy as np

def auc(fpr: list, tpr: list) -> float:
    """
    Returns the area under the ROC curve as a float.
    """

    # Convert lists into NumPy arrays
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)

    # Calculate AUC using trapezoidal rule
    area = np.trapezoid(tpr, fpr)

    return float(area)


# Example
fpr = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
tpr = [0.0, 0.5, 0.7, 0.8, 0.9, 1.0]

result = auc(fpr, tpr)

print("AUC:", result)
