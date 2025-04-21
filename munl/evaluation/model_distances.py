import numpy as np
import torch
from torch import Tensor
from torch.nn import Module


# NOTE: Adapted from https://github.com/AdityaGolatkar/SelectiveForgetting/blob/master/Forgetting.ipynb
# function `distance`
def compute_model_distance(model, model0):
    for p in model.parameters():
        p.data0 = p.data.clone()
    for p in model0.parameters():
        p.data0 = p.data.clone()
    distance = 0
    normalization = 0
    for (k, p), (k0, p0) in zip(model.named_parameters(), model0.named_parameters()):
        current_dist = (p.data0 - p0.data0).pow(2).sum().item()
        current_norm = p.data0.pow(2).sum().item()
        distance += current_dist
        normalization += current_norm
    print(f"Distance: {np.sqrt(distance)}")
    print(f"Normalized Distance: {1.0*np.sqrt(distance/normalization)}")
    return 1.0 * np.sqrt(distance / normalization)


def models_l2_distance(model1, model2):
    """
    Calculate the L2 distance (Euclidean distance) between the weights of two models.

    Args:
    model1: A PyTorch model.
    model2: A PyTorch model, should have the same architecture as model1.

    Returns:
    float: The L2 distance between the model weights.
    """
    distance = 0.0
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if (
            p1.data.nelement() == p2.data.nelement()
        ):  
            diff = p1.data - p2.data
            distance += torch.norm(diff, p=2) ** 2
        else:
            raise ValueError(
                "Models have different architectures or parameter configurations."
            )

    return torch.sqrt(distance).item()


def model_l2_norm(model: Module) -> float:
    """Compute the L2 norm of the model's parameters.

    Args:
        model (Module): Model to compute from

    Returns:
        float: Total L2 Norm
    """
    distance = torch.zeros(1)
    for param in model.parameters():
        distance += torch.norm(param, p=2) ** 2
    return torch.sqrt(distance).item()


def models_normalized_l2_distance(model_a: Module, model_b: Module) -> float:
    """Compute the L2 norm of the difference between two normalized models' parameters."""
    norm_a = model_l2_norm(model_a)
    norm_b = model_l2_norm(model_b)

    distance = torch.zeros(1)
    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        normalized_param_a = param_a / norm_a
        normalized_param_b = param_b / norm_b
        distance += torch.norm(normalized_param_a - normalized_param_b, p=2) ** 2
    return torch.sqrt(distance).item()
