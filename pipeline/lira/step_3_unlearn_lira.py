#!/usr/bin/env python
import logging
import os
import pathlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch

# from hydra_zen import make_config, hydrated_dataclass, zen
from hydra.conf import HydraConf, JobConf, RunDir
from hydra.core.hydra_config import HydraConfig
from hydra_zen import make_config, store, zen
from torch.utils.tensorboard import SummaryWriter

from munl.configurations import (
    DatasetConfig,
    ModelConfig,
    UnlearnerConfig,
    get_img_size_for_dataset,
)
from munl.models import get_model_from_cfg
from munl.settings import DEFAULT_RANDOM_STATE
from pipeline.optuna_utils import get_loaders
from pipeline.step_5_unlearn import UnlearnerApp

logger = logging.getLogger(__name__)


store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"dataset": "cifar10"},
            {"model": "resnet18"},
            {"unlearner": "naive"},
        ],
        unlearner=None,
        model=None,
        dataset=None,
        model_seed=0,
        random_state=DEFAULT_RANDOM_STATE,
        save_path=None,
        split_ndx=0,
        forget_ndx=0,
    ),
    name="unlearn_lira_model",
)


def zen_unlearn_lira_model(
    dataset: DatasetConfig,
    model: ModelConfig,
    unlearner: UnlearnerConfig,
    model_seed: int,
    random_state: int,
    save_path=None,
    writer=None,
    save_steps: bool = False,
    should_evaluate: bool = False,
    split_ndx: int = 0,
    forget_ndx: int = 0,
):
    if writer == "tensorboard":
        writer = SummaryWriter()
    unlearner.save_steps = save_steps
    unlearner.writer = writer
    unlearner.should_evaluate = should_evaluate

    lira = True
    root = pathlib.Path(hydra.utils.get_original_cwd())
    app = UnlearnerApp(
        dataset_cfg=dataset,
        unlearner=unlearner,
        model_cfg=model,
        model_seed=model_seed,
        random_state=random_state,
    )
    print("split_ndx", split_ndx, "forget_ndx", forget_ndx)
    assert (split_ndx is not None and forget_ndx is not None) or (
        split_ndx is None and forget_ndx is None
    )
    assert lira and split_ndx is not None and forget_ndx is not None
    save_name = f"{split_ndx}_{forget_ndx}"

    model_cfg = model
    dataset_cfg = dataset
    dataset_name = dataset_cfg.name

    img_size = get_img_size_for_dataset(dataset_name)
    root = pathlib.Path(hydra.utils.get_original_cwd())
    unlearner.cfg.model_initializations_dir = "unlearn/original"
    weights_root = root / "artifacts" / dataset_name / "artifacts" / "lira"
    name_save_path = root / "artifacts" / dataset_name / "lira" / "unlearn"

    model = get_model_from_cfg(
        root=weights_root,
        model_cfg=model_cfg,
        unlearner_cfg=unlearner.cfg,
        num_classes=dataset_cfg.num_classes,
        model_seed=model_seed,
        img_size=img_size,
        split_ndx=split_ndx,
        forget_ndx=forget_ndx,
    )

    app = UnlearnerApp(
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        unlearner=unlearner,
        model_seed=model_seed,
        random_state=random_state,
    )
    train_loader, retain_loader, forget_loader, val_loader, test_loader = get_loaders(
        root=root,
        dataset_cfg=dataset_cfg,
        unlearner_cfg=unlearner.cfg,
        lira=lira,
        split_ndx=split_ndx,
        forget_ndx=forget_ndx,
        random_state=random_state,
    )
    original_model, unlearned_model = app.run_from_model_and_loaders(
        root=root,
        model=model,
        loaders=[train_loader, retain_loader, forget_loader, val_loader, test_loader],
        save=True,
        save_path=name_save_path,
        save_name=save_name,
        unlearner_name=None,
    )
    ###


def generate_dir_name():
    pbs_index = os.environ.get("PBS_ARRAY_INDEX", "")
    now = datetime.now()
    dir_name = now.strftime("outputs/%Y-%m-%d/%H-%M-%S")
    save_name = f"{dir_name}/{pbs_index}" if pbs_index else dir_name
    save_name = f"{save_name}/{uuid.uuid4()}"
    return save_name


if __name__ == "__main__":
    store(
        HydraConf(job=JobConf(chdir=True), run=RunDir(dir=generate_dir_name())),
        name="config",
        group="hydra",
    )
    store.add_to_hydra_store()
    zen(zen_unlearn_lira_model).hydra_main(
        config_name="unlearn_lira_model",
        version_base="1.1",
        config_path=None,
    )
