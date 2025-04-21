import pathlib
import typing as typ
from enum import Enum

import numpy as np
import sklearn.model_selection as skmos
from numpy import array as Array


def assert_no_overlap(first: set, second: set):
    """Assertion chek for non overlapping sets

    Args:
        first (set): First set to consider
        second (set): Second set to consider
    """
    assert set(first).isdisjoint(set(second))


def assert_is_subset(subset: set, superset: set):
    """Assertion check for subset

    Args:
        subset (set): Set that needs to be a subset
        superset (set): Set that represents the superset
    """
    assert set(subset).issubset(set(superset))


def generate_splits_indices(
    train_size: int,
    test_size: int,
    val_ratio: float,
    forget_ratio: float,
    random_state: int,
) -> typ.Tuple[Array, Array, Array, Array, Array]:
    """Genearte the different splits of the dataset

    Args:
        train_size (int): number of elements in the training set
        test_size (int): number of entries in the test set
        val_ratio (float): ratio [0, 1] from the training set to use as validation set
        forget_ratio (float): ration [0, 1] from the training set to use as forget set
        random_state (int): Random seed used to generate the splits

    Returns:
        typ.Tuple[Array, Array, Array, Array, Array]: Indices for the (train, val, retain, forget, test) splits
    """
    assert 0.0 <= val_ratio and val_ratio <= 1.0
    assert 0.0 <= forget_ratio and forget_ratio <= 1.0
    total_size = train_size + test_size
    total_indices = Array(range(total_size))
    dev_indices = total_indices[:train_size]
    test_indices = total_indices[train_size:]
    assert set(dev_indices).isdisjoint(set(test_indices))
    train_indices, val_indices = skmos.train_test_split(
        dev_indices, test_size=val_ratio, random_state=random_state
    )
    retain_indices, forget_indices = skmos.train_test_split(
        train_indices, test_size=forget_ratio, random_state=random_state
    )
    # Train + Val + Test must not overlap
    assert_no_overlap(train_indices, val_indices)
    # Retain and Forget must be part of Train
    assert_is_subset(retain_indices, train_indices)
    assert_is_subset(forget_indices, train_indices)
    # Retain and Forget must not overlap
    assert_no_overlap(retain_indices, forget_indices)
    return train_indices, retain_indices, forget_indices, val_indices, test_indices


def save_split(
    split: np.ndarray,
    output_path: pathlib.Path,
    dataset_name: str,
    split_name: str,
    random_state: int,
):
    """Save the split to disk"""
    output_split_path = get_output_split_path(
        output_path, dataset_name, split_name, random_state
    )
    output_split_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_split_path, split)


def get_output_split_path(
    output_dir: pathlib.Path, dataset_name: str, split_name: str, random_state: int
) -> pathlib.Path:
    """Get the path to save the split to"""
    return output_dir / dataset_name / str(random_state) / f"{split_name}.npy"
