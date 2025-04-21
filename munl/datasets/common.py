import typing as typ

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset

from munl.settings import DEFAULT_PIN_MEMORY, DEFAULT_RANDOM_STATE
from munl.utils import get_num_workers_from_shuffle


def is_shuffling(dataloader: DataLoader) -> bool:
    """
    Check if the DataLoader is shuffling the data.

    Parameters:
    dataloader (DataLoader): The DataLoader to check.

    Returns:
    bool: True if the DataLoader is shuffling, False otherwise.
    """
    return isinstance(dataloader.sampler, RandomSampler)


def update_dataloader_batch_size(
    existing_dataloader: DataLoader, new_batch_size: int
) -> DataLoader:
    """
    Updates the batch size of an existing DataLoader.

    Parameters:
    existing_dataloader (DataLoader): The original DataLoader.
    new_batch_size (int): The new batch size.

    Returns:
    DataLoader: A new DataLoader with the updated batch size.
    """
    dataset = existing_dataloader.dataset
    shuffle = is_shuffling(existing_dataloader)
    num_workers = existing_dataloader.num_workers
    collate_fn = existing_dataloader.collate_fn
    pin_memory = existing_dataloader.pin_memory
    drop_last = existing_dataloader.drop_last
    timeout = existing_dataloader.timeout
    worker_init_fn = existing_dataloader.worker_init_fn
    multiprocessing_context = existing_dataloader.multiprocessing_context

    new_dataloader = DataLoader(
        dataset,
        batch_size=new_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=multiprocessing_context,
    )

    return new_dataloader


class PlaceHolderDataset(Dataset):
    """Dataset to simulate a dataset with a given number of samples"""

    def __init__(self, num_samples: int, data_shape=(3, 32, 32), num_classes: int = 10):
        """Base Constructor

        Args:
            num_samples (int): Number of samples to simulate in the dataset
            data_shape (tuple, optional): Shape of the data to generate. Defaults to (3, 32, 32).
            num_classes (int, optional): Number of classes to consider. Defaults to 10.
        """
        self.num_samples = num_samples
        self.data_shape = data_shape
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, _) -> typ.Tuple[Tensor, int]:
        image = torch.randn(size=self.data_shape)
        target = np.random.choice(np.arange(self.num_classes), size=1)[0]
        return image, target


class ManualDataset(Dataset):
    def __init__(self, data: Tensor, targets: Tensor):
        self.data = data
        self.targets = targets
        assert len(self.data) == len(
            self.targets
        ), "Data and targets should have the same length"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, ndx) -> typ.Tuple[Tensor, typ.Any]:
        return self.data[ndx], self.targets[ndx]


def equalize_datasets(datasets: typ.List[Dataset]) -> typ.List[Dataset]:
    """Given multipled datasets, return subsets of similar size

    Args:
        datasets (typ.List[Dataset]): _description_

    Returns:
        typ.List[Dataset]: _description_
    """
    assert isinstance(datasets, list), "Datasets need to be provided as a list"
    lengths = np.array([len(dataset) for dataset in datasets])
    min_pos = np.argmin(lengths)
    min_length = lengths[min_pos]
    equalized = []
    for dataset in datasets:
        equalized.append(
            Subset(dataset, np.random.choice(len(dataset), min_length, replace=False))
        )
    return equalized


class DiscernibleCombinedDataset(Dataset):
    def __init__(self, first_dataset: Dataset, second_dataset: Dataset):
        self.first_dataset = first_dataset
        self.second_dataset = second_dataset
        self.total_len = len(self.first_dataset) + len(self.second_dataset)

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, index) -> typ.Tuple[Tensor, typ.Any, int]:
        """Getter for the dataset

        Args:
            index (_type_): Index of the element to get

        Returns:
            typ.Tuple[Tensor, typ.Any, int]: Get the origina sample, its label and the origin of the sample
        """
        if index < len(self.first_dataset):
            data, label = self.first_dataset[index]
            origin = 0
        else:
            data, label = self.second_dataset[index - len(self.first_dataset)]
            origin = 1
        return data, label, origin


def get_combined_retain_and_forget_loaders(
    retain_loader: DataLoader, forget_loader: DataLoader, shuffle: bool
) -> DataLoader:
    batch_size = retain_loader.batch_size
    assert (
        batch_size == forget_loader.batch_size
    ), "Retain and Forget loaders batch sizes must be the same."
    retain_dataset = retain_loader.dataset
    forget_dataset = forget_loader.dataset
    combined_dataset = torch.utils.data.ConcatDataset([retain_dataset, forget_dataset])
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=get_num_workers_from_shuffle(shuffle),
        pin_memory=DEFAULT_PIN_MEMORY,
    )
    assert len(combined_dataset) == len(retain_dataset) + len(
        forget_dataset
    ), "Combined loader must have the same number of batches as the sum of the retain and forget loaders."
    return combined_loader


def get_discernible_retain_and_forget_loaders(
    retain_loader: DataLoader, forget_loader: DataLoader, shuffle: bool
) -> DataLoader:
    batch_size = retain_loader.batch_size
    assert (
        batch_size == forget_loader.batch_size
    ), "Retain and Forget loaders batch sizes must be the same."
    retain_dataset = retain_loader.dataset
    forget_dataset = forget_loader.dataset
    combined_dataset = DiscernibleCombinedDataset(retain_dataset, forget_dataset)
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=get_num_workers_from_shuffle(shuffle),
        pin_memory=DEFAULT_PIN_MEMORY,
    )
    assert len(combined_loader.dataset) == len(retain_loader.dataset) + len(
        forget_loader.dataset
    ), "Combined loader must have the same number of batches as the sum of the retain and forget loaders."
    return combined_loader


class RandomRelabelDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        num_classes: int,
        random_state: int = DEFAULT_RANDOM_STATE,
    ):
        self.original_dataset = dataset
        self.num_classes = num_classes
        self.rng = np.random.RandomState(random_state)

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, target = self.original_dataset[idx]
        target = self.rng.randint(0, self.num_classes)
        return image, target


def extract_targets_only(dataloader: DataLoader, dtype=int) -> Tensor:
    """Get the targets from a DataLoader

    Args:
        dataloader (DataLoader): DataLoader to extract the targets from
        dtype (_type_, optional): Type the targetrs. Defaults to int.

    Returns:
        Tensor: resulting target tensor
    """
    assert is_shuffling(dataloader) is False, "Dataloader should not shuffle"
    targets = []
    for _, target in dataloader:
        targets.extend(target.numpy())
    return torch.as_tensor(targets, dtype=dtype)
