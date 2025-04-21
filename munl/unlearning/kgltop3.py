import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from munl.datasets import get_combined_retain_and_forget_loaders
from munl.models import get_optimizer_scheduler_criterion
from munl.settings import DEFAULT_MODEL_INIT_DIR
from munl.unlearning.common import BaseUnlearner, train_one_epoch


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(CustomCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, input, target):
        ce_loss = nn.functional.cross_entropy(input, target)

        if self.class_weights is not None:
            weights = torch.tensor(
                [self.class_weights[i] for i in target], device=input.device
            )
            ce_loss = torch.mean(ce_loss * weights)

        return ce_loss


def vision_confuser(model, std=0.6):
    for name, module in model.named_children():
        if hasattr(module, "weight"):
            if "conv" in name:
                actual_value = module.weight.clone().detach()
                new_values = torch.normal(mean=actual_value, std=std)
                module.weight.data.copy_(new_values)


import munl.settings
from munl.evaluation.common import extract_predictions
from munl.settings import HP_NORMAL_SIGMA


class KGLTop3(BaseUnlearner):
    ORIGINAL_NUM_EPOCHS = 4
    ORIGINAL_W = 0.05
    ORIGINAL_LR = 0.0007
    ORIGINAL_MOMENTUM = 0.9
    ORIGINAL_WEIGHT_DECAY = 5e-4
    ORIGINAL_START_CONFUSER_STD = 0.6
    ORIGINAL_FINAL_CONFUSER_STD = 0.005
    ORIGINAL_BATCH_SIZE = 64
    ORIGINAL_INIT_RATE = 0.3

    HYPER_PARAMETERS = {
        "unlearner.cfg.start_confuser_std": HP_NORMAL_SIGMA,
        "unlearner.cfg.final_confuser_std": HP_NORMAL_SIGMA,
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
        import numpy as np
        import sklearn.utils

        device = self.device
        net.to(device)

        # Define the vision_confuser function
        start_confuser_std = self.cfg.start_confuser_std
        final_confuser_std = self.cfg.final_confuser_std
        vision_confuser(net, std=start_confuser_std)

        epochs = self.cfg.num_epochs

        w = self.cfg.class_weight
        lr = self.cfg.optimizer.learning_rate
        momentum = self.cfg.optimizer.momentum
        weight_decay = self.cfg.optimizer.weight_decay
        targets, _ = extract_predictions(net, retain_loader, device)  #
        unique_targets = np.unique(targets)
        class_weights = sklearn.utils.class_weight.compute_class_weight(
            class_weight="balanced", classes=unique_targets, y=targets
        )

        # class_weights = [1, w, w, w, w, w, w, w, w, w] original Value

        criterion = CustomCrossEntropyLoss(class_weights)

        optimizer = optim.SGD(
            net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        net.train()

        for ep in range(epochs):
            net.train()
            for sample in retain_loader:
                inputs, targets = sample
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            if ep == epochs - 2:
                vision_confuser(
                    net, std=final_confuser_std
                )  # increase model robustness before last training epoch

            scheduler.step()

        net.eval()
        return net


import typing as typ


def kgl3_default_optimizer():
    return {
        # "type": "torch.optim.SGD",
        "learning_rate": KGLTop3.ORIGINAL_LR,
        "momentum": KGLTop3.ORIGINAL_MOMENTUM,
        "weight_decay": KGLTop3.ORIGINAL_WEIGHT_DECAY,
    }


import typing as typ
from dataclasses import dataclass, field

from munl.settings import default_loaders
from munl.utils import DictConfig


@dataclass
class DefaultKGLTop3Config:
    num_epochs: int = KGLTop3.ORIGINAL_NUM_EPOCHS
    batch_size: int = KGLTop3.ORIGINAL_BATCH_SIZE
    class_weight: float = KGLTop3.ORIGINAL_W
    start_confuser_std: float = KGLTop3.ORIGINAL_START_CONFUSER_STD
    final_confuser_std: float = KGLTop3.ORIGINAL_FINAL_CONFUSER_STD

    optimizer: typ.Dict[str, typ.Any] = field(default_factory=kgl3_default_optimizer)
    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)
