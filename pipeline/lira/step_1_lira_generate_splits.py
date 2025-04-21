import copy
from argparse import Namespace
from collections import Counter
from pathlib import Path
import argparse
import numpy as np

from munl.datasets import get_dataset_and_lengths
from munl.datasets.cifar10 import (
    get_cifar10_train_transform,
)


CIFAR10 = get_dataset_and_lengths(
    Path("datasets"), dataset_name="cifar10", transform=get_cifar10_train_transform()
)
DATASET, (TRAIN_LEN, TEST_LEN) = CIFAR10


def generate_all_forgets(train_matrices, num_attempts, ratio):
    retains, forgets = [], []
    for row in train_matrices:
        retain, forget = generate_lira_train_tests(row, num_attempts, ratio)
        retains.append(retain)
        forgets.append(forget)
    retains = np.stack(retains)
    forgets = np.stack(forgets)
    return retains, forgets


def get_retain_forget_val_test_indices(
    lira_path: Path, split_ndx: int, forget_ndx: int, verbose: bool = True
):
    retains = np.load(lira_path / str(split_ndx) / "retains.npy")
    forgets = np.load(lira_path / str(split_ndx) / "forgets.npy")
    vals = np.load(lira_path / "val_matrices.npy")
    tests = np.load(lira_path / "test_matrices.npy")
    assert 0 <= forget_ndx < len(retains)
    assert retains.ndim == 2
    assert forgets.ndim == 2
    retain_indices = retains[forget_ndx]
    forget_indices = forgets[forget_ndx]
    val_indices = vals
    test_indices = tests[split_ndx]
    assert retain_indices.ndim == 1
    assert forget_indices.ndim == 1
    assert val_indices.ndim == 1
    if verbose:
        print(retain_indices, forget_indices, val_indices, test_indices)

    assert set(retain_indices) & set(forget_indices) == set()
    assert set(retain_indices) & set(test_indices) == set()
    assert set(forget_indices) & set(test_indices) == set()
    assert set(val_indices) & set(test_indices) == set()
    assert set(retain_indices) & set(val_indices) == set()
    assert set(forget_indices) & set(val_indices) == set()
    assert set(test_indices) & set(val_indices) == set()
    return retain_indices, forget_indices, val_indices, test_indices


def generate_lira_train_tests(lira_dev_indices, num_attempts, ratio=0.5):
    test_size = int(len(lira_dev_indices) * ratio)
    train_size = len(lira_dev_indices) - test_size
    indices = np.zeros(shape=(num_attempts, len(lira_dev_indices)), dtype=int)
    # assert train_size == test_size
    for attempt_ndx in range(num_attempts):
        indices[attempt_ndx] = np.random.default_rng(attempt_ndx).permutation(
            copy.deepcopy(lira_dev_indices)
        )
    train_indices, test_indices = indices[:, :train_size], indices[:, train_size:]
    # We verify that the indices row by row are different
    for train_row, test_row in zip(train_indices, test_indices):
        assert len(set(train_row) & set(test_row)) == 0
    return train_indices, test_indices


def obtain_counts(indices, ax):
    counter = Counter(indices.ravel())
    values = list(counter.values())
    ax.hist(values, bins=100)
    return np.mean(values), np.std(values)


def main(args):
    data_seed = args.data_seed
    val_ratio = args.val_ratio
    forget_ratio = args.forget_ratio
    train_test_attempts = args.train_tests_attempts
    retain_forget_attempts = args.retain_forget_attempts
    cifar10 = get_dataset_and_lengths(
        Path("datasets"),
        dataset_name="cifar10",
        transform=get_cifar10_train_transform(),
    )
    dataset, (train_len, test_len) = cifar10
    indices = np.arange((len(dataset)))
    dev_indices = indices[:train_len]
    test_indices = indices[train_len:]
    assert len(dev_indices) == train_len
    assert len(test_indices) == test_len
    val_len = int(train_len * val_ratio)

    print(np.random.default_rng(data_seed).permutation(dev_indices))
    lira_indices = np.random.default_rng(data_seed).permutation(dev_indices)
    lira_dev_indices = lira_indices[:-val_len]
    lira_val_indices = lira_indices[-val_len:]
    assert len(lira_dev_indices) == train_len - val_len
    assert len(lira_val_indices) == val_len

    train_matrices, test_matrices = generate_lira_train_tests(
        lira_dev_indices=lira_dev_indices, num_attempts=train_test_attempts
    )
    path = Path("artifacts/lira/splits")
    if not path.exists():
        path.mkdir(parents=True)
    np.save(path / "train_matrices.npy", train_matrices)
    np.save(path / "val_matrices.npy", lira_val_indices)
    np.save(path / "test_matrices.npy", test_matrices)

    retains, forgets = generate_all_forgets(
        train_matrices, num_attempts=retain_forget_attempts, ratio=forget_ratio
    )
    assert retains.shape[0] == train_test_attempts
    for split_ndx in range(train_test_attempts):
        split_path = path / str(split_ndx)
        if not split_path.exists():
            split_path.mkdir(parents=True)
        np.save(split_path / "retains.npy", retains[split_ndx])
        np.save(split_path / "forgets.npy", forgets[split_ndx])
        print(
            f"Saved split {(retains[split_ndx].shape, forgets[split_ndx].shape)}  [{split_ndx + 1} / {train_test_attempts}]"
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_seed", type=int, default=123)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--forget_ratio", type=float, default=0.1)
    parser.add_argument("--train_tests_attempts", type=int, default=64)
    parser.add_argument("--retain_forget_attempts", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
