#!/usr/bin/env python
import logging
from pathlib import Path

from hydra_zen import make_config, store

from munl import DEFAULT_FORGET_RATIO, DEFAULT_VAL_RATIO, FIXED_SPLITS_PATH
from munl.configurations import DatasetConfig
from munl.datasets import get_dataset_and_lengths
from munl.datasets.splits import generate_splits_indices, save_split
from munl.settings import DEFAULT_RANDOM_STATE
from munl.utils import DataSplit, setup_seed

logger = logging.getLogger(__name__)

store(
    make_config(
        hydra_defaults=["_self_", {"dataset": "cifar10"}],
        dataset=None,
        datasets_path="datasets",
        output_path=FIXED_SPLITS_PATH,
        val_ratio=DEFAULT_VAL_RATIO,
        forget_ratio=DEFAULT_FORGET_RATIO,
        random_state=DEFAULT_RANDOM_STATE,
    ),
    name="generate_fixed_splits",
)


def zen_generate_fixed_splits(
    datasets_path: str,
    dataset: DatasetConfig,
    output_path: str,
    val_ratio: float,
    forget_ratio: float,
    random_state: int,
):
    return generate_fixed_splits(
        Path(datasets_path),
        dataset.name,
        Path(output_path),
        val_ratio,
        forget_ratio,
        random_state,
    )


def generate_fixed_splits(
    datasets_path: Path,
    dataset_name: str,
    output_path: Path,
    val_ratio: float,
    forget_ratio: float,
    random_state: int,
):
    # For reproducibility
    setup_seed(random_state)

    _, lengths = get_dataset_and_lengths(
        datasets_root=datasets_path, dataset_name=dataset_name, transform=None
    )
    assert (
        len(lengths) == 2
    ), "There should be two splits in the dataset. (Training and Test set)"
    train_size, test_size = lengths
    num_samples = train_size + test_size
    (
        train_indices,
        retain_indices,
        forget_indices,
        val_indices,
        test_indices,
    ) = generate_splits_indices(
        train_size=train_size,
        test_size=test_size,
        val_ratio=val_ratio,
        forget_ratio=forget_ratio,
        random_state=random_state,
    )
    logger.info(f"Saving splits for dataset {dataset_name} ({num_samples})")
    for split, split_name in zip(
        [train_indices, retain_indices, forget_indices, val_indices, test_indices],
        [
            DataSplit.train,
            DataSplit.retain,
            DataSplit.forget,
            DataSplit.val,
            DataSplit.test,
        ],
    ):
        save_split(
            split=split,
            output_path=output_path,
            dataset_name=dataset_name,
            split_name=split_name,
            random_state=random_state,
        )
        logger.info(
            f"Saved {split_name} split for {dataset_name} dataset ({len(split)})"
        )


if __name__ == "__main__":
    from hydra.conf import HydraConf, JobConf
    from hydra_zen import zen

    store(HydraConf(job=JobConf(chdir=False)), name="config", group="hydra")
    store.add_to_hydra_store()
    zen(zen_generate_fixed_splits).hydra_main(
        config_name="generate_fixed_splits", version_base="1.1", config_path=None
    )
