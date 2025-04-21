from torch.utils.data import DataLoader, RandomSampler

from munl.datasets.cifar10 import CIFAR10_IMAGE_SIZE
from munl.datasets.cifar100 import CIFAR100_IMAGE_SIZE
from munl.datasets.fashion_mnist import FASHION_MNIST_IMAGE_SIZE
from munl.datasets.mnist import MNIST_IMAGE_SIZE
from munl.datasets.utkface import UTKFACE_IMAGE_SIZE, UTKFace

from .common import (
    DiscernibleCombinedDataset,
    ManualDataset,
    PlaceHolderDataset,
    RandomRelabelDataset,
    equalize_datasets,
    extract_targets_only,
    get_combined_retain_and_forget_loaders,
    get_discernible_retain_and_forget_loaders,
    is_shuffling,
    update_dataloader_batch_size,
)
from .get_dataset import (
    DATASET_NAME_TO_TORCHVISION,
    get_dataset_and_lengths,
    get_loaders_from_dataset_and_unlearner_from_cfg,
    get_loaders_from_dataset_and_unlearner_from_cfg_with_indices,
    is_train_from_data_split,
)
