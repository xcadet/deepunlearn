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


class KGLTop5(BaseUnlearner):
    ORIGINAL_NUM_EPOCHS = 3
    ORIGINAL_LR = 0.001
    ORIGINAL_MOMENTUM = 0.9
    ORIGINAL_WEIGHT_DECAY = 5e-4
    ORIGINAL_BATCH_SIZE = 64

    HYPER_PARAMETERS = {**munl.settings.HYPER_PARAMETERS}

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
        device = self.device
        net.to(device)

        epochs = self.cfg.num_epochs
        lr = self.cfg.optimizer.learning_rate
        momentum = self.cfg.optimizer.momentum
        weight_decay = self.cfg.optimizer.weight_decay

        def rotate_weight(local_model):
            print("rotate weight")
            for module in local_model.modules():
                if isinstance(module, torch.nn.modules.conv.Conv2d):
                    module.weight = torch.nn.Parameter(module.weight.swapaxes(2, 3))

        rotate_weight(net)

        print(
            "Criterion, opimizer and scheduler are defined as provided by the authors."
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        net.train()

        for ep in range(epochs):
            for sample in retain_loader:
                inputs, targets = sample
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            scheduler.step()

        net.eval()
        return net


def kgl5_default_optimizer():
    return {
        "type": "torch.optim.SGD",
        "learning_rate": KGLTop5.ORIGINAL_LR,
        "momentum": KGLTop5.ORIGINAL_MOMENTUM,
        "weight_decay": KGLTop5.ORIGINAL_WEIGHT_DECAY,
    }


import typing as typ
from dataclasses import dataclass, field

from munl.settings import DEFAULT_MODEL_INIT_DIR, default_loaders
from munl.utils import DictConfig


@dataclass
class DefaultKGLTop5Config:
    num_epochs: int = KGLTop5.ORIGINAL_NUM_EPOCHS
    batch_size: int = KGLTop5.ORIGINAL_BATCH_SIZE

    optimizer: typ.Dict[str, typ.Any] = field(default_factory=kgl5_default_optimizer)
    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)
