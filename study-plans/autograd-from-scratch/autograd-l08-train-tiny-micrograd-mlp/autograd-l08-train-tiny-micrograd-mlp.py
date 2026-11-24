
import torch

def train_tiny_micrograd_mlp(inputs: torch.Tensor, targets: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor], learning_rate: float, steps: int) -> tuple:
    """
    Returns final predictions, final loss, trained weights, trained biases, and pre-update loss history.
    """

    # Clone parameters to avoid modifying original inputs
    weights = [w.clone() for w in weights]
    biases = [b.clone() for b in biases]

    loss_history = []

    def forward(x):
        activations = [x]

        for w, b in zip(weights, biases):
            z = activations[-1] @ w.T + b
            h = torch.tanh(z)
            activations.append(h)

        return activations

    # Perform exactly the requested number of updates
    for _ in range(steps):

        # Forward propagation
        activations = forward(inputs)
        predictions = activations[-1].squeeze(-1)

        # Summed squared error
        errors = predictions - targets
        loss = torch.sum(errors ** 2)

        # Record pre-update loss
        loss_history.append(loss.clone())

        # Manual backpropagation
        delta = (2.0 * errors).unsqueeze(1)

        weight_grads = [None] * len(weights)
        bias_grads = [None] * len(biases)

        for i in reversed(range(len(weights))):

            # Derivative of tanh
            delta = delta * (1.0 - activations[i + 1] ** 2)

            # Calculate gradients
            weight_grads[i] = delta.T @ activations[i]
            bias_grads[i] = torch.sum(delta, dim=0)

            # Propagate gradient to previous layer
            if i > 0:
                delta = delta @ weights[i]

        # Simultaneous gradient descent update
        new_weights = []
        new_biases = []

        for i in range(len(weights)):
            new_weights.append(
                weights[i] - learning_rate * weight_grads[i]
            )

            new_biases.append(
                biases[i] - learning_rate * bias_grads[i]
            )

        weights = new_weights
        biases = new_biases

    # Calculate final predictions and loss
    with torch.no_grad() if False else torch.enable_grad():
        pass

    activations = [inputs]

    for w, b in zip(weights, biases):
        activations.append(torch.tanh(activations[-1] @ w.T + b))

    predictions = activations[-1].squeeze(-1)
    final_loss = torch.sum((predictions - targets) ** 2)

    return predictions, final_loss, weights, biases, loss_history
