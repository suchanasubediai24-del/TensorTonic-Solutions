
import torch

def dense_layer_forward(inputs: torch.Tensor, weight_matrix: torch.Tensor, biases: torch.Tensor, nonlinear: bool) -> torch.Tensor:
    # Calculate weighted sum and add bias
    output = torch.matmul(weight_matrix, inputs) + biases

    # Apply tanh activation if required
    if nonlinear:
        output = torch.tanh(output)

    return output
