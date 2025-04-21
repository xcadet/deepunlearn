import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from munl import DEFAULT_DEVICE


def compute_losses(
    net: nn.Module, loader: DataLoader, device: torch.device = DEFAULT_DEVICE
) -> np.ndarray:
    """Auxiliary function to compute per-sample losses"""

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = net(inputs)
        losses = criterion(logits, targets).numpy(force=True)
        all_losses.extend(losses.tolist())
    return np.array(all_losses)
