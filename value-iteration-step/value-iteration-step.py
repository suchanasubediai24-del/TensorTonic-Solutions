
def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    """
    Returns one updated floating-point value for every state.
    """
    num_states = len(values)
    new_values = []

    for s in range(num_states):
        best_value = float('-inf')

        for a in range(len(rewards[s])):
            expected_value = 0.0

            for next_state in range(num_states):
                expected_value += (
                    transitions[s][a][next_state] * values[next_state]
                )

            q_value = rewards[s][a] + gamma * expected_value

            best_value = max(best_value, q_value)

        new_values.append(float(best_value))

    return new_values
