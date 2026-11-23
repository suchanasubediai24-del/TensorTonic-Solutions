
def gae(rewards: list, values: list, gamma: float, lam: float) -> list:
    """
    Returns the generalized advantage estimate at every timestep.
    """

    # Get number of timesteps
    T = len(rewards)

    # Initialize advantages with zeros
    advantages = [0.0] * T

    # Initialize next advantage
    next_advantage = 0.0

    # Calculate advantages backward
    for t in range(T - 1, -1, -1):

        # Calculate TD error
        delta = rewards[t] + gamma * values[t + 1] - values[t]

        # Calculate GAE
        advantages[t] = delta + gamma * lam * next_advantage

        # Update next advantage
        next_advantage = advantages[t]

    return advantages
