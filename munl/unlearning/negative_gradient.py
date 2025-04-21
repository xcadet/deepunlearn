import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

import munl.settings
from munl.datasets import get_discernible_retain_and_forget_loaders
from munl.models import get_optimizer_scheduler_criterion
from munl.unlearning.common import BaseUnlearner


def train_one_epoch_negative_gradients(
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
        losses = criterion(outputs, targets)
        losses[forget_set_mask] = -losses[forget_set_mask]
        random_chance = 1 / outputs.shape[1]
        # Clamp to chance of random guessing
        clamped_losses = torch.clamp(losses, min=0, max=random_chance)
        loss = clamped_losses.mean()
        batch_losses[batch_ndx] = loss.detach().item()
        loss.backward()
        optimizer.step()
    scheduler.step()
    return batch_losses


class NegativeGradient(BaseUnlearner):
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
            scheduler,
            criterion,
        ) = get_optimizer_scheduler_criterion(model, self.cfg)
        model.to(device)
        # We avoid the criterion from the config because we need to use the loss over each sample
        training_criterion = nn.CrossEntropyLoss(reduction="none")
        train_loader = get_discernible_retain_and_forget_loaders(
            retain_loader, forget_loader, shuffle=True
        )
        for epoch in tqdm(range(self.cfg.num_epochs)):
            train_batch_loss = train_one_epoch_negative_gradients(
                model, train_loader, optimizer, scheduler, training_criterion, device
            )
            val_batch_loss = self.evaluate_if_needed(
                model, val_loader, criterion, device
            )
            payload = {
                "train_loss": train_batch_loss.mean(),
                "val_loss": val_batch_loss.mean(),
            }
            self.save_and_log(model, optimizer, scheduler, payload, epoch)

        return model
