# Adapted from: https://github.com/OPTML-Group/Unlearn-Sparse/blob/public/unlearn/GA.py
import typing as typ
from dataclasses import dataclass, field

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

import munl.settings as settings
from munl.models import get_optimizer_scheduler_criterion
from munl.settings import DEFAULT_MODEL_INIT_DIR, default_loaders
from munl.unlearning.common import BaseUnlearner
from munl.utils import DictConfig



class GradientAscent(BaseUnlearner):
    ORIGINAL_LR = 0.01
    ORIGINAL_MOMENTUM = 0.9
    ORIGINAL_WEIGHT_DECAY = 5e-4
    ORIGINAL_NUM_EPOCHS = 10
    ORIGINAL_BATCH_SIZE = 256

    HYPER_PARAMETERS = {
        **settings.HYPER_PARAMETERS,
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
        device = self.device
        (
            optimizer,
            scheduler,
            _,
        ) = get_optimizer_scheduler_criterion(model, self.cfg)
        criterion = torch.nn.CrossEntropyLoss()
        model.to(device)
        for epoch in range(self.cfg.num_epochs):
            model = GA(model, forget_loader, criterion, optimizer, device)
        return model


def gradient_ascent_default_optimizer():
    return {
        "type": "torch.optim.SGD",
        "learning_rate": GradientAscent.ORIGINAL_LR,
        "momentum": GradientAscent.ORIGINAL_MOMENTUM,
        "weight_decay": GradientAscent.ORIGINAL_WEIGHT_DECAY,
    }


@dataclass
class GradientAscentConfig:
    num_epochs: int = GradientAscent.ORIGINAL_NUM_EPOCHS
    batch_size: int = GradientAscent.ORIGINAL_BATCH_SIZE

    optimizer: typ.Dict[str, typ.Any] = field(
        default_factory=gradient_ascent_default_optimizer
    )
    scheduler: typ.Union[typ.Dict[str, typ.Any], None] = None
    criterion: typ.Union[typ.Dict[str, typ.Any], None] = None
    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def GA(
    model: Module,
    forget_loader: DataLoader,
    criterion: Module,
    optimizer,
    device: str,
) -> Module:
    model.train()
    print(len(forget_loader.dataset))

    for i, (image, target) in enumerate(forget_loader):
        image = image.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        output_clean = model(image)
        loss = -criterion(output_clean, target)

        loss.backward()
        optimizer.step()

    return model