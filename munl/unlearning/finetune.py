import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

import munl.settings
from munl.models import get_optimizer_scheduler_criterion
from munl.unlearning.common import BaseUnlearner, train_one_epoch


class FinetuneUnlearner(BaseUnlearner):
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
        for epoch in tqdm(range(self.cfg.num_epochs)):
            train_batch_loss = train_one_epoch(
                model, retain_loader, optimizer, scheduler, criterion, device
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
