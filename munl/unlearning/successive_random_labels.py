import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

import munl.settings
from munl.datasets import get_discernible_retain_and_forget_loaders
from munl.models import get_optimizer_scheduler_criterion
from munl.unlearning.common import BaseUnlearner
from munl.unlearning.salun import RandomRelabel
from munl.utils import get_num_classes_from_model


def unlearn_one_epoch_random_labels(
    model, train_loader, optimizer, scheduler, criterion, device
):
    model = model.train()
    num_batches = len(train_loader)
    batch_losses = torch.zeros(num_batches)
    for batch_ndx, (inputs, targets, origin) in enumerate(train_loader):
        inputs, targets, origin = (
            inputs.to(device),
            targets.to(device),
            origin.to(device),
        )
        forget_set_mask = origin == 1
        optimizer.zero_grad()
        outputs = model(inputs)
        num_classes = outputs.shape[1]
        targets[forget_set_mask] = torch.randint_like(
            targets[forget_set_mask], 0, num_classes
        )
        loss = criterion(outputs, targets)
        batch_losses[batch_ndx] = loss.detach().item()
        loss.backward()
        optimizer.step()
    scheduler.step()
    return batch_losses


class SuccessiveRandomLabels(BaseUnlearner):
    HYPER_PARAMETERS = munl.settings.HYPER_PARAMETERS

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
        model: nn.Module,
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        val_loader: DataLoader,
    ) -> nn.Module:
        device = self.device
        (
            optimizer,
            _,
            criterion,
        ) = get_optimizer_scheduler_criterion(model, self.cfg)
        batch_size = self.cfg.batch_size
        num_classes = get_num_classes_from_model(model)
        mask = None

        model.to(device)
        for _ in tqdm(range(self.cfg.num_epochs)):
            model = RandomRelabel(
                model,
                retain_loader=retain_loader,
                forget_loader=forget_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_classes=num_classes,
                batch_size=batch_size,
                mask=mask,
                device=device,
            )
        return model
