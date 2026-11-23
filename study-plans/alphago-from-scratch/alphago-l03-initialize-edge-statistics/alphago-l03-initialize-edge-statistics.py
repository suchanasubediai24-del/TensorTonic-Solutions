
import numpy as np

def initialize_mcts_edges(policy_logits: np.ndarray, legal_mask: np.ndarray) -> dict:
    """
    Initialize MCTS edge statistics.

    Returns:
        N: int64 array (visit counts)
        W: float64 array (total action values)
        Q: float64 array (mean action values)
        P: float64 array (prior probabilities)
    """

    policy_logits = np.asarray(policy_logits, dtype=np.float64)
    legal_mask = np.asarray(legal_mask, dtype=bool)

    # Initialize arrays
    N = np.zeros(len(policy_logits), dtype=np.int64)
    W = np.zeros(len(policy_logits), dtype=np.float64)
    Q = np.zeros(len(policy_logits), dtype=np.float64)
    P = np.zeros(len(policy_logits), dtype=np.float64)

    # Select legal actions
    if np.any(legal_mask):
        legal_logits = policy_logits[legal_mask]

        # Numerically stable softmax
        exp_logits = np.exp(legal_logits - np.max(legal_logits))

        # Normalize probabilities
        P[legal_mask] = exp_logits / np.sum(exp_logits)

    return {
        "N": N,
        "W": W,
        "Q": Q,
        "P": P
    }

