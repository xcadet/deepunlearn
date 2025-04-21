from typing import Optional
from munl.datasets import get_loaders_from_dataset_and_unlearner_from_cfg
from pathlib import Path
import numpy as np
from munl.datasets import (
    get_loaders_from_dataset_and_unlearner_from_cfg_with_indices,
)
from pipeline.lira.step_1_lira_generate_splits import get_retain_forget_val_test_indices


def get_loaders(
    root: Path,
    dataset_cfg,
    unlearner_cfg,
    lira: bool,
    split_ndx: Optional[int],
    forget_ndx: Optional[int],
    random_state=123,
):

    if lira:
        print("Loading LiRA split.")
        retain_indices, forget_indices, val_indices, test_indices = (
            get_retain_forget_val_test_indices(
                lira_path=root / "artifacts" / "lira" / "splits",
                split_ndx=split_ndx,
                forget_ndx=forget_ndx,
            )
        )
        train_indices = np.concatenate([retain_indices, forget_indices])
        train_loader, retain_loader, forget_loader, val_loader, test_loader = (
            get_loaders_from_dataset_and_unlearner_from_cfg_with_indices(
                root=root,
                indices=[
                    train_indices,
                    retain_indices,
                    forget_indices,
                    val_indices,
                    test_indices,
                ],
                dataset_cfg=dataset_cfg,
                unlearner_cfg=unlearner_cfg,
            )
        )
        print(
            f"Loaded LiRA split: {len(train_indices)} train, {len(retain_indices)} retain, {len(forget_indices)} forget, {len(val_indices)} val, {len(test_indices)} test."
        )
    else:
        (
            train_loader,
            retain_loader,
            forget_loader,
            val_loader,
            test_loader,
        ) = get_loaders_from_dataset_and_unlearner_from_cfg(
            root=root,
            dataset_cfg=dataset_cfg,
            unlearner_cfg=unlearner_cfg,
            random_state=random_state,
        )
    return train_loader, retain_loader, forget_loader, val_loader, test_loader
