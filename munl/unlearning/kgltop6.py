# Code adapted from https://www.kaggle.com/code/stathiskaripidis/unlearning-by-resetting-layers-7th-on-private-lb
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from munl.unlearning.common import BaseUnlearner


def kl_loss(model_logits, teacher_logits, temperature=1.0):
    """
    Calculate the Kullback-Leibler (KL) divergence loss between the output of a model (student) and the output of a teacher model.

    Args:
        model_logits (torch.Tensor): The logits from the student model. Logits are the raw outputs of the last neural network layer prior to softmax activation.
        teacher_logits (torch.Tensor): The logits from the teacher model.
        temperature (float, optional): A temperature parameter that scales the logits before applying softmax. Higher temperatures produce softer probability distributions. Defaults to 1.

    Returns:
        torch.Tensor: The KL divergence loss, averaged over the batch.
    """
    # apply softmax to the (scaled) logits of the teacher model
    teacher_output_softmax = F.softmax(teacher_logits / temperature, dim=1)
    # apply log softmax to the (scaled) logits of the student model
    output_log_softmax = F.log_softmax(model_logits / temperature, dim=1)

    # calculate the KL divergence between the student and teacher outputs
    kl_div = F.kl_div(output_log_softmax, teacher_output_softmax, reduction="batchmean")
    return kl_div


def soft_cross_entropy(preds, soft_targets):
    """
    Calculate the soft cross-entropy loss between predictions and soft targets.

    Args:
        preds (torch.Tensor): The predictions from the student model.
        soft_targets (torch.Tensor): The soft targets, the probability distribution from a teacher model.

    Returns:
        torch.Tensor: The average soft cross-entropy loss across all instances in the batch.
    """
    # calculate the element-wise cross-entropy loss
    loss = torch.sum(-soft_targets * torch.log_softmax(preds, dim=1), dim=1)
    return torch.mean(loss)


import munl.settings


class KGLTop6(BaseUnlearner):
    # WARMUP
    ORIGINAL_WARMUP_EPOCHS = 3
    ORIGINAL_WARMUP_LR = 9e-4
    ORIGINAL_WARMUP_MOMENTUM = 0.9
    ORIGINAL_WARMUP_WEIGHT_DECAY = 5e-4

    ORIGINAL_RETAIN_EPOCHS = 3
    ORIGINAL_RETAIN_LR = 1e-3
    ORIGINAL_RETAIN_MOMENTUM = 0.9
    ORIGINAL_RETAIN_WEIGHT_DECAY = 5e-4

    ORIGINAL_TEMPERATURE = 5.0
    ORIGINAL_BATCH_SIZE = 64

    HYPER_PARAMETERS = {
        "unlearner.cfg.temperature": munl.settings.HP_TEMPERATURE,
        "unlearner.cfg.retain_num_epochs": munl.settings.HP_NUM_EPOCHS,
        "unlearner.cfg.retain_optimizer.learning_rate": munl.settings.HP_LEARNING_RATE,
        "unlearner.cfg.retain_optimizer.momentum": munl.settings.HP_MOMENTUM,
        "unlearner.cfg.retain_optimizer.weight_decay": munl.settings.HP_WEIGHT_DECAY,
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
        model,
        retain_loader,
        forget_loader,
        val_loader,
    ):
        device = self.device
        model.to(device)
        teacher_model = copy.deepcopy(model)
        if hasattr(
            model, "conv1"
        ):  
            model.conv1.reset_parameters()
        if hasattr(model, "fc"):
            model.fc.reset_parameters()
        elif hasattr(model, "head"):
            model.head.reset_parameters()
        else:
            raise NotImplementedError("The model does not have a fc or head attribute")

        # set the teacher model to evaluation mode
        teacher_model.eval()

        # define the number of epochs for warm-up and retain phases

        warmup_epochs = self.cfg.num_epochs
        warmup_lr = self.cfg.optimizer.learning_rate
        warmup_weigth_decay = self.cfg.optimizer.weight_decay
        warmup_momentum = self.cfg.optimizer.momentum

        retain_epochs = self.cfg.retain_num_epochs
        retain_lr = self.cfg.retain_optimizer.learning_rate
        retain_weigth_decay = self.cfg.retain_optimizer.weight_decay
        retain_momentum = self.cfg.retain_optimizer.momentum

        temperature = self.cfg.temperature

        # standard cross-entropy loss for fine-tuning
        criterion = nn.CrossEntropyLoss()

        # Warm-up phase: Adjust the student model closer to the teacher model using knowledge distillation
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=warmup_lr,
            weight_decay=warmup_weigth_decay,
            momentum=warmup_momentum,
        )
        for epoch in range(warmup_epochs):
            model.train()
            for sample in val_loader:
                x, y = sample
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                student_out = model(x)

                with torch.no_grad():
                    teacher_out = teacher_model(x)

                loss = kl_loss(
                    model_logits=student_out,
                    teacher_logits=teacher_out,
                    temperature=temperature,
                )
                loss.backward()
                optimizer.step()

        # Fine-tuning phase: Train the model on the retain set using standard cross-entropy along with knowledge distillation
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=retain_lr,
            weight_decay=retain_weigth_decay,
            momentum=retain_momentum,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=retain_epochs
        )
        for epoch in range(retain_epochs):
            model.train()
            for sample in retain_loader:
                x, y = sample
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                with torch.no_grad():
                    teacher_out = teacher_model(x)
                soft_labels = torch.softmax(teacher_out / temperature, dim=1)
                soft_predictions = torch.log_softmax(out / temperature, dim=1)
                loss = soft_cross_entropy(soft_predictions, soft_labels)
                loss += criterion(out, y)
                loss += kl_loss(
                    model_logits=out,
                    teacher_logits=teacher_out,
                    temperature=temperature,
                )
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        return model


def kgl6_default_optimizer():
    return {
        "type": "torch.optim.SGD",
        "learning_rate": KGLTop6.ORIGINAL_WARMUP_LR,
        "momentum": KGLTop6.ORIGINAL_WARMUP_MOMENTUM,
        "weight_decay": KGLTop6.ORIGINAL_WARMUP_WEIGHT_DECAY,
    }


def kgl6_default_retain_optimizer():
    return {
        "type": "torch.optim.SGD",
        "learning_rate": KGLTop6.ORIGINAL_RETAIN_LR,
        "momentum": KGLTop6.ORIGINAL_RETAIN_MOMENTUM,
        "weight_decay": KGLTop6.ORIGINAL_RETAIN_WEIGHT_DECAY,
    }


import typing as typ
from dataclasses import dataclass, field

from munl.settings import DEFAULT_MODEL_INIT_DIR, default_loaders
from munl.utils import DictConfig


@dataclass
class DefaultKGLTop6Config:
    num_epochs: int = KGLTop6.ORIGINAL_WARMUP_EPOCHS
    retain_num_epochs: int = KGLTop6.ORIGINAL_RETAIN_EPOCHS
    batch_size: int = KGLTop6.ORIGINAL_BATCH_SIZE
    temperature: float = KGLTop6.ORIGINAL_TEMPERATURE

    optimizer: typ.Dict[str, typ.Any] = field(default_factory=kgl6_default_optimizer)
    retain_optimizer: typ.Dict[str, typ.Any] = field(
        default_factory=kgl6_default_retain_optimizer
    )
    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)
