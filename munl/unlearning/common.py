import importlib
import pathlib
import typing as typ
from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from munl.utils import DictConfig, get_num_workers_from_shuffle


def instantiate_optimizer(optimizer_spec, model_parameters):
    module_name, class_name = optimizer_spec["type"].rsplit(".", 1)
    module = importlib.import_module(module_name)
    optimizer_class = getattr(module, class_name)

    optimizer_spec["params"] = model_parameters

    optimizer = optimizer_class(**optimizer_spec)

    return optimizer


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    criterion: nn.Module,
    device: torch.device,
) -> Tensor:
    model = model.train()
    num_batches = len(train_loader)
    batch_losses = torch.zeros(num_batches)
    for batch_ndx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        batch_losses[batch_ndx] = loss.detach().item()
        loss.backward()
        optimizer.step()
    scheduler.step()
    return batch_losses


def evaluate(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tensor:
    model = model.eval()
    num_bacthes = len(val_loader)
    batch_losses = torch.zeros(num_bacthes)
    for batch_ndx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        batch_losses[batch_ndx] = loss.detach().item()
    return batch_losses


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch: int,
    payload: typ.Dict[str, typ.Any],
    filename: str = "checkpoint.pth",
    save_dir: str = "checkpoints",
) -> None:
    print("Saving checkpoint.")
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "payload": payload,
    }
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / filename
    torch.save(state, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    filename: str = "checkpoint.pth",
) -> typ.Tuple[nn.Module, Optimizer, LRScheduler, int, typ.Dict[str, typ.Any]]:
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    epoch = checkpoint["epoch"]
    payload = checkpoint["payload"]
    return model, optimizer, scheduler, epoch, payload


class BaseUnlearner:
    def __init__(
        self,
        cfg: DictConfig,
        device,
        writer=None,
        save_steps: bool = False,
        should_evaluate: bool = False,
    ):
        self.cfg = cfg
        self.writer = writer
        self.device = device
        self.save_steps = save_steps
        self.should_evaluate = should_evaluate

    @abstractmethod
    def unlearn(
        self,
        model: nn.Module,
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        val_loader: DataLoader,
    ) -> nn.Module:
        """
        Unlearns the model on the forget_loader and then retrains it on the retain_loader.
        :param model: The model to unlearn.
        :param retain_loader: The data to retain.
        :param forget_loader: The data to forget.
        :param val_loader: The data to validate on.
        :return: The unlearned model.
        """

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(save={self.save_steps}, evaluate={self.should_evaluate}): {self.cfg}"

    def save_and_log(self, model, optimizer, scheduler, payload, epoch):
        if self.writer is not None:
            self.writer.add_scalars("log", payload, epoch)
        if self.save_steps:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                payload=payload,
                filename=f"epoch_{epoch}.pth",
            )

    def evaluate_if_needed(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> Tensor:
        """If the unlearner should evaluate, then evaluate the model on the val_loader and return the loss.
        Otherwise, return a tensor of zeros."""
        val_batch_loss = torch.zeros(len(val_loader))
        if self.should_evaluate:
            val_batch_loss = evaluate(model, val_loader, criterion, device)
        return val_batch_loss
