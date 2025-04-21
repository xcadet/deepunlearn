import hashlib
import logging
import os
import random
import typing as typ
import thirdparty.tiny_vit as tiny_vit
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from pandas import DataFrame
from torch import Tensor
from torch.nn import Module

from munl.models import format_model_path

LOGGER = logging.getLogger(__name__)


class AutoValueEnum(Enum):
    def __str__(self):
        return str(self.value)


class DataSplit(AutoValueEnum):
    train = "train"
    val = "val"
    retain = "retain"
    forget = "forget"
    test = "test"


class TransformState(AutoValueEnum):
    augmented = "augmented"
    unmodified = "unmodified"


def get_allocated_cpus():
    # For Slurm clusters
    if "SLURM_CPUS_ON_NODE" in os.environ:
        return int(os.environ["SLURM_CPUS_ON_NODE"])
    # For PBS clusters
    elif "PBS_NP" in os.environ:
        return int(os.environ["PBS_NP"])
    else:
        return os.cpu_count()


def get_num_workers_from_shuffle(
    shuffle: bool, default: typ.Union[int, None] = None
) -> int:
    """If we are shuffling the data we do not neccesarily need to use multiple workers."""
    if default is None:
        default = int(get_allocated_cpus() // 2)
    return default


def get_save_path(
    root: Path,
    dataset_name: str,
    num_classes: int,
    step_name: str,
    model_name: str,
    model_seed: int,
    img_size: int,
    overrides: typ.Union[typ.List[str], None] = None,
) -> Path:
    overrides = HydraConfig.get().overrides.task if overrides is None else overrides
    filtered_overrides = sorted(
        [
            override
            for override in overrides
            if not override.startswith("model_seed=")
            and not override.startswith("dataset=")
        ]
    )

    overrides_str = "-".join(filtered_overrides).replace("/", "_").replace("=", "_")

    artifacts_path = root / "artifacts"
    filename = Path(
        format_model_path(
            str(artifacts_path / dataset_name / step_name / f"{overrides_str}"),
            num_classes,
            model_name=model_name,
            seed=model_seed,
            img_size=img_size,
        )
    )
    return filename


def get_num_classes_from_model(model: Module) -> int:
    import torchvision.models

    if isinstance(model, torchvision.models.ResNet):
        return model.fc.out_features
    elif isinstance(model, tiny_vit.TinyViT):
        return model.head.out_features
    else:
        raise ValueError(f"Model type {type(model)} not supported.")


def is_conv2d(module: nn.Module) -> bool:
    return isinstance(module, nn.Conv2d)


def is_linear(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear)


class DictConfig:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictConfig(value))
            else:
                setattr(self, key, value)

    def __str__(self):
        def stringify(obj):
            if isinstance(obj, DictConfig):
                return str(obj)
            else:
                return repr(obj)

        items = [f"{key}: {stringify(value)}" for key, value in self.__dict__.items()]
        return "{" + ", ".join(items) + "}"


def convert_int_or_list_to_nparray(
    to_convert: typ.Union[int, typ.List[int]]
) -> np.ndarray:
    if isinstance(to_convert, int):
        to_convert = [to_convert]
    elif isinstance(to_convert, Tensor):
        to_convert = to_convert.detach().cpu()
    array = np.array(to_convert)
    return array


def setup_seed(seed):
    LOGGER.info(f"setup random seed = {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def interleave_outputs(first: Tensor, second: Tensor, origin: Tensor) -> Tensor:
    assert first.shape == second.shape
    assert first.ndim == second.ndim == 2
    assert origin.ndim == 1
    assert origin.dtype == torch.int64
    assert origin.shape[0] == first.shape[0]
    assert set(origin.unique().tolist()).issubset(set([0, 1]))

    outputs = torch.zeros_like(first)
    is_forget = origin == 1
    outputs[~is_forget] = first[~is_forget]
    outputs[is_forget] = second[is_forget]
    return outputs


def create_or_update_symlinks(source_dir, target_dir):
    """
    Creates or updates symlinks in the target directory to match the source directory structure and files.
    If the source item is a directory, a symlink to the directory will be created instead of a physical directory.

    Args:
    - source_dir (Path or str): The source directory containing the original files and directories.
    - target_dir (Path or str): The target directory where symlinks will be created or updated.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    # Ensure the target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over each item in the source directory
    for item in source_dir.iterdir():
        # Define the target path for the symlink in the target directory
        target_path = target_dir / item.name

        # Check if a symlink already exists at the target path
        if target_path.exists():
            if target_path.is_symlink():
                # Resolve the absolute path of the symlink's target and the intended target
                actual_target = target_path.resolve(strict=False)
                intended_target = item.resolve()

                # Check if the symlink points to the correct target
                if actual_target != intended_target:
                    print(
                        f"Symlink exists but points to a different target: {target_path} -> {actual_target}"
                    )
                    # Update the symlink to point to the correct target
                    target_path.unlink()  # Remove the incorrect symlink
                    target_path.symlink_to(
                        intended_target, target_is_directory=item.is_dir()
                    )  # Recreate the symlink correctly
                    print(f"Updated symlink: {target_path} -> {intended_target}")
                else:
                    print(
                        f"Correct symlink already exists: {target_path} -> {actual_target}"
                    )
            else:
                print(f"Target exists but is not a symlink: {target_path}")
        else:
            # Create a symlink for both files and directories
            target_path.symlink_to(item.resolve(), target_is_directory=item.is_dir())
            print(f"Created symlink: {target_path} -> {item.resolve()}")

    print("Symlinking operation completed.")


def compute_md5(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def successive_join(dfs: List[DataFrame]) -> DataFrame:
    """Join multiple dataframes based on their index

    Args:
        dfs (List[DataFrame]): List of dataframes to join

    Returns:
        DataFrame: The joined dataframe
    """
    df = dfs[0]
    for new_df in dfs[1:]:
        df = df.join(new_df, how="outer")
    return df


def extract_list_of_ints(string: str) -> list[int]:
    return list(map(int, string.split(",")))


def extract_list_of_strings(string: str) -> list[str]:
    return string.split(",")
