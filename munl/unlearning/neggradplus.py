import copy
import typing as typ
from dataclasses import dataclass, field
from itertools import cycle

import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

import munl.settings
import munl.settings as settings
from munl.datasets import RandomRelabelDataset
from munl.models import get_optimizer_scheduler_criterion
from munl.settings import DEFAULT_DEVICE, DEFAULT_MODEL_INIT_DIR, default_loaders
from munl.unlearning.common import BaseUnlearner
from munl.unlearning.scrub_utils import l2_penalty
from munl.utils import DictConfig, get_num_classes_from_model


def run_neggrad_epoch(
    model: Module,
    model_init: nn.Module,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    alpha: float,
    criterion: Module,
    optimizer: torch.optim.Optimizer,
    weight_decay: float,
):
    model.eval()

    with torch.set_grad_enabled(True):
        for _, (batch_retain, batch_forget) in enumerate(
            zip(retain_loader, cycle(forget_loader))
        ):
            batch_retain = [
                tensor.to(next(model.parameters()).device) for tensor in batch_retain
            ]
            batch_forget = [
                tensor.to(next(model.parameters()).device) for tensor in batch_forget
            ]
            input_r, target_r = batch_retain
            input_f, target_f = batch_forget
            output_r = model(input_r)
            output_f = model(input_f)
            loss = alpha * (
                criterion(output_r, target_r)
                + l2_penalty(model, model_init, weight_decay)
            ) - (1 - alpha) * criterion(output_f, target_f)
            model.zero_grad()
            loss.backward()
            optimizer.step()
    return model


# NOTE: Adapted from https://github.com/meghdadk/SCRUB/blob/main/small_scale_unlearning.ipynb
# 1. Adding optimizer
def negative_grad(
    model: Module,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    criterion: Module,
    optimizer: torch.optim.Optimizer,
    weight_decay: float,
    alpha: float,
    epochs: int,
):
    model_init = copy.deepcopy(model)
    for epoch in range(epochs):
        model = run_neggrad_epoch(
            model=model,
            model_init=model_init,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            alpha=alpha,
            criterion=criterion,
            optimizer=optimizer,
            weight_decay=weight_decay,
        )
    return model


class NegGradPlus(BaseUnlearner):
    # Specific to the original implementation

    # Generic Hyper
    ORIGINAL_LR = 0.01
    ORIGINAL_MOMENTUM = 0.9
    ORIGINAL_WEIGHT_DECAY = 0.0
    ORIGINAL_NUM_EPOCHS = 10
    ORIGINAL_BATCH_SIZE = 256
    ORIGINAL_ALPHA = 0.95

    HYPER_PARAMETERS = {
        **settings.HYPER_PARAMETERS,
        "unlearner.cfg.alpha": munl.settings.HP_FLOAT,
    }

    def __init__(
        self,
        cfg: DictConfig,
        device,
        writer=None,
        save_steps: bool = False,
        should_evaluate: bool = False,
    ):
        super().__init__(
            cfg,
            device=device,
            writer=writer,
            save_steps=save_steps,
            should_evaluate=should_evaluate,
        )

    def unlearn(
        self,
        model: Module,
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Module:
        # Variables preparation
        device = self.device
        epochs = self.cfg.num_epochs
        alpha = self.cfg.alpha
        weight_decay = self.cfg.weight_decay
        # Optimizer, Scheduler and Criterion
        (
            optimizer,
            _,
            _,
        ) = get_optimizer_scheduler_criterion(model, self.cfg)
        criterion = torch.nn.CrossEntropyLoss()

        model.to(device)
        model = negative_grad(
            model=model,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            criterion=criterion,
            optimizer=optimizer,
            weight_decay=weight_decay,
            alpha=alpha,
            epochs=epochs,
        )
        return model


def neggradplus_default_optimizer():
    return {
        "type": "torch.optim.SGD",
        "learning_rate": NegGradPlus.ORIGINAL_LR,
        "momentum": NegGradPlus.ORIGINAL_MOMENTUM,
        "weight_decay": 0.0,
    }


@dataclass
class DefaultNegGradPlusUnlearningConfig:
    num_epochs: int = NegGradPlus.ORIGINAL_NUM_EPOCHS
    batch_size: int = NegGradPlus.ORIGINAL_BATCH_SIZE
    alpha: float = NegGradPlus.ORIGINAL_ALPHA
    weight_decay = NegGradPlus.ORIGINAL_WEIGHT_DECAY

    optimizer: typ.Dict[str, typ.Any] = field(
        default_factory=neggradplus_default_optimizer
    )
    scheduler: typ.Union[typ.Dict[str, typ.Any], None] = None
    criterion: typ.Union[typ.Dict[str, typ.Any], None] = None
    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)
