import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Performs one RMSProp optimization step.

    Parameters:
        w: Current weights
        g: Current gradients
        s: Running average of squared gradients
        lr: Learning rate
        beta: Decay rate
        eps: Small value to avoid division by zero

    Returns:
        (new_w, new_s)
    """

    # Convert lists to NumPy arrays
    w = np.array(w, dtype=float)
    g = np.array(g, dtype=float)
    s = np.array(s, dtype=float)

    # Update running average of squared gradients
    new_s = beta * s + (1 - beta) * (g ** 2)

    # Update weights using RMSProp
    new_w = w - lr * g / (np.sqrt(new_s) + eps)

    # Convert back to lists
    return new_w.tolist(), new_s.tolist()