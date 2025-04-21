import typing as typ
from pathlib import Path

import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import munl.visualization as vis


def extract_age_from_filename(filename: str) -> int:
    content = filename.split("_")
    return int(content[0])


class UTKFace(Dataset):
    EXPECTED_NUM_IMAGES = 23708
    DEFAULT_TEST_RATIO = 0.2
    DEFAULT_RANDOM_SEED = 123

    def __init__(self, root: Path, train: bool, transform=None):
        super().__init__()
        root = Path(root) / "UTKface_Aligned_cropped" / "UTKFace"
        self.transform = transform
        self.train = train
        images = np.array(
            list(filter(lambda path: path.suffix == ".jpg", root.iterdir()))
        )
        names = [image.name.split(".")[0] for image in images]
        ages = np.array([extract_age_from_filename(name) for name in names])
        targets = np.floor(ages / 20).astype(int)  # 0-indexed
        targets = np.clip(targets, a_min=0, a_max=4)
        assert len(np.unique(targets)) == 5
        assert all(targets >= 0) and all(targets < 5)

        indices = np.arange(len(images))
        permuted_indices = np.random.default_rng(self.DEFAULT_RANDOM_SEED).permutation(
            indices
        )
        test_size = int(self.DEFAULT_TEST_RATIO * len(images))
        train_indices = permuted_indices[test_size:]
        test_indices = permuted_indices[:test_size]
        if train:
            self.targets = targets[train_indices]
            self.images = images[train_indices]
        else:
            self.targets = targets[test_indices]
            self.images = images[test_indices]
        assert len(self.images) == len(
            self.targets
        ), f"Number of images ({len(self.images)}) and targets ({len(self.targets)}) do not match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ndx):
        img = Image.open(self.images[ndx])
        if self.transform is not None:
            img = self.transform(img)
        target = self.targets[ndx]
        return img, target

    def __repr__(self):
        return f"UTKFace Dataset (size={len(self)}, transform={self.transform})"


UTKFACE_IMAGE_SIZE = 224
UTKFACE_MEAN = [0.485, 0.456, 0.406]
UTKFACE_STD = [0.229, 0.224, 0.225]


def get_utkface_train_transform():
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                size=(UTKFACE_IMAGE_SIZE, UTKFACE_IMAGE_SIZE),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=UTKFACE_MEAN, std=UTKFACE_STD),
        ]
    )
    return train_transform


def get_utkface_test_transform():
    return get_utkface_train_transform()
