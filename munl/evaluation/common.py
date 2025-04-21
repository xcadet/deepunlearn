import typing as typ

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_dtypes(dataset) -> typ.Tuple[typ.Any, typ.Any]:
    """Obtain the data types from the dataset"""
    print(dataset)
    data, targets = dataset[0]
    if isinstance(targets, int) or (
        isinstance(targets, torch.Tensor) and targets.dtype == torch.int64
    ):
        targets = np.array([targets], dtype=int)
    return data.dtype, targets.dtype


def extract_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> typ.Tuple[np.ndarray, np.ndarray]:
    """Extract the true labels and predictions from a model and a data loader

    Args:
        model (nn.Module): Model that outputs logits
        loader (DataLoader): Dataloader to obtain predictions from
        device (torch.device): Device to compute on

    Returns:
        typ.Tuple[np.ndarray, np.ndarray]: (True labels, Predicted labels)
    """
    model.eval()
    num_entries = len(loader.dataset)
    _, targets_dtype = get_dtypes(loader.dataset)
    predictions = np.zeros(num_entries, dtype=targets_dtype)
    y_true = np.zeros(num_entries, dtype=targets_dtype)
    batch_size = loader.batch_size
    for batch_ndx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        start = batch_ndx * batch_size
        end = start + batch_size
        predictions[start:end] = model(images).argmax(dim=1).cpu().numpy()
        y_true[start:end] = targets.cpu().numpy()
    return y_true, predictions


def extract_target_and_outputs(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> typ.Tuple[np.ndarray, np.ndarray]:
    """Extract the true labels and predictions from a model and a data loader

    Args:
        model (nn.Module): Model that outputs logits
        loader (DataLoader): Dataloader to obtain predictions from
        device (torch.device): Device to compute on

    Returns:
        typ.Tuple[np.ndarray, np.ndarray]: (True labels, Predicted labels)
    """
    model.eval()
    num_entries = len(loader.dataset)
    _, targets_dtype = get_dtypes(loader.dataset)
    output_size = model((loader.dataset[0][0]).unsqueeze(0).to(device)).shape[-1]
    predictions = np.zeros(shape=(num_entries, output_size))
    y_true = np.zeros(num_entries, dtype=targets_dtype)
    batch_size = loader.batch_size
    for batch_ndx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        start = batch_ndx * batch_size
        end = start + batch_size
        predictions[start:end] = model(images).cpu().numpy()
        y_true[start:end] = targets.cpu().numpy()
    return y_true, predictions
