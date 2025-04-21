import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from munl.datasets import get_combined_retain_and_forget_loaders
from munl.models import get_optimizer_scheduler_criterion
from munl.unlearning.common import BaseUnlearner, train_one_epoch


def kl_loss_sym(x, y):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    return kl_loss(nn.LogSoftmax(dim=-1)(x), y)


import munl.settings


class KGLTop1(BaseUnlearner):
    ORIGINAL_NUM_EPOCHS = 8
    ORIGINAL_BATCH_SIZE = 256

    ORIGINAL_LR = 0.005
    ORIGINAL_MOMENTUM = 0.9
    ORIGINAL_WEIGHT_DECAY = 0

    ORIGINAL_RETAIN_LR = 0.001
    ORIGINAL_RETAIN_MOMENTUM = 0.9
    ORIGINAL_RETAIN_WEIGHT_DECAY = 1e-2

    ORIGINAL_FORGET_LR = 3e-4
    ORIGINAL_FORGET_MOMENTUM = 0.9
    ORIGINAL_FORGET_WEIGHT_DECAY = 0

    ORIGINAL_ETA_MIN = 1e-6

    ORIGINAL_TEMPERATURE = 1.15

    HYPER_PARAMETERS = {
        "unlearner.cfg.retain_optimizer.learning_rate": munl.settings.HP_LEARNING_RATE,
        "unlearner.cfg.retain_optimizer.momentum": munl.settings.HP_MOMENTUM,
        "unlearner.cfg.retain_optimizer.weight_decay": munl.settings.HP_WEIGHT_DECAY,
        "unlearner.cfg.forget_optimizer.learning_rate": munl.settings.HP_LEARNING_RATE,
        "unlearner.cfg.forget_optimizer.momentum": munl.settings.HP_MOMENTUM,
        "unlearner.cfg.forget_optimizer.weight_decay": munl.settings.HP_WEIGHT_DECAY,
        "unlearner.cfg.temperature": munl.settings.HP_TEMPERATURE,
        "unlearner.cfg.eta_min": munl.settings.HP_ETA_MIN,
        **munl.settings.HYPER_PARAMETERS,
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
        net,
        retain_loader,
        forget_loader,
        val_loader,
    ):
        """Simple unlearning by finetuning."""
        print("-----------------------------------")
        net.to(self.device)
        device = self.device
        epochs = self.cfg.num_epochs
        batch_size = (
            self.cfg.batch_size
        )  
        criterion = nn.CrossEntropyLoss()

        opt_lr = self.cfg.optimizer.learning_rate
        opt_mom = self.cfg.optimizer.momentum
        opt_weight_decay = self.cfg.optimizer.weight_decay

        opt_retain_lr = self.cfg.retain_optimizer.learning_rate * batch_size / 64
        opt_retain_mom = self.cfg.retain_optimizer.momentum
        opt_retain_weight_decay = self.cfg.retain_optimizer.weight_decay

        opt_forget_lr = self.cfg.forget_optimizer.learning_rate
        opt_forget_momentum = self.cfg.forget_optimizer.momentum
        opt_forget_weight_decay = self.cfg.forget_optimizer.weight_decay

        temperature = self.cfg.temperature
        eta_min = self.cfg.eta_min

        optimizer = optim.SGD(
            net.parameters(), lr=opt_lr, momentum=opt_mom, weight_decay=opt_weight_decay
        )
        optimizer_retain = optim.SGD(
            net.parameters(),
            lr=opt_retain_lr,
            momentum=opt_retain_mom,
            weight_decay=opt_retain_weight_decay,
        )
        optimizer_forget = optim.SGD(
            net.parameters(),
            lr=opt_forget_lr,
            momentum=opt_forget_momentum,
            weight_decay=opt_forget_weight_decay,
        )
        total_step = int(len(forget_loader) * epochs)
        retain_ld = DataLoader(
            retain_loader.dataset, batch_size=batch_size, shuffle=True
        )
        retain_ld4fgt = DataLoader(
            retain_loader.dataset, batch_size=batch_size, shuffle=True
        )
        scheduler = CosineAnnealingLR(
            optimizer_forget, T_max=total_step, eta_min=eta_min
        )
        net.train()
        for sample in forget_loader:  ##First Stage
            inputs = sample[0]
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            uniform_label = (
                torch.ones_like(outputs).to(device) / outputs.shape[1]
            )  ##uniform pseudo label
            loss = kl_loss_sym(
                outputs, uniform_label
            )  ##optimize the distance between logits and pseudo labels
            loss.backward()
            optimizer.step()
        net.train()
        for ep in range(epochs):  ##Second Stage
            net.train()
            for sample_forget, sample_retain in zip(
                forget_loader, retain_ld4fgt
            ):  ##Forget Round
                inputs_forget, inputs_retain = (sample_forget[0], sample_retain[0])
                inputs_forget, inputs_retain = inputs_forget.to(
                    device
                ), inputs_retain.to(device)
                optimizer_forget.zero_grad()
                outputs_forget, outputs_retain = (
                    net(inputs_forget),
                    net(inputs_retain).detach(),
                )
                loss = (
                    -1
                    * nn.LogSoftmax(dim=-1)(
                        outputs_forget @ outputs_retain.T / temperature
                    )
                ).mean()  ##Contrastive Learning loss
                loss.backward()
                optimizer_forget.step()
                scheduler.step()
            for sample in retain_ld:  ##Retain Round
                inputs, labels = sample
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer_retain.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_retain.step()
        print("-----------------------------------")
        return net


def kgl1_default_optimizer():
    return {
        "learning_rate": KGLTop1.ORIGINAL_LR,
        "momentum": KGLTop1.ORIGINAL_MOMENTUM,
        "weight_decay": KGLTop1.ORIGINAL_WEIGHT_DECAY,
    }


def kgl1_default_retain_optimizer():
    return {
        "learning_rate": KGLTop1.ORIGINAL_RETAIN_LR,
        "momentum": KGLTop1.ORIGINAL_RETAIN_MOMENTUM,
        "weight_decay": KGLTop1.ORIGINAL_RETAIN_WEIGHT_DECAY,
    }


def kgl1_default_forget_optimizer():
    return {
        "learning_rate": KGLTop1.ORIGINAL_FORGET_LR,
        "momentum": KGLTop1.ORIGINAL_FORGET_MOMENTUM,
        "weight_decay": KGLTop1.ORIGINAL_FORGET_WEIGHT_DECAY,
    }


import typing as typ
from dataclasses import dataclass, field

from munl.settings import DEFAULT_MODEL_INIT_DIR, default_loaders
from munl.utils import DictConfig


@dataclass
class DefaultKGLTop1Config:
    num_epochs: int = KGLTop1.ORIGINAL_NUM_EPOCHS
    batch_size: int = KGLTop1.ORIGINAL_BATCH_SIZE
    temperature: float = KGLTop1.ORIGINAL_TEMPERATURE
    eta_min: float = KGLTop1.ORIGINAL_ETA_MIN

    optimizer: typ.Dict[str, typ.Any] = field(default_factory=kgl1_default_optimizer)
    retain_optimizer: typ.Dict[str, typ.Any] = field(
        default_factory=kgl1_default_retain_optimizer
    )
    forget_optimizer: typ.Dict[str, typ.Any] = field(
        default_factory=kgl1_default_forget_optimizer
    )
    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)
