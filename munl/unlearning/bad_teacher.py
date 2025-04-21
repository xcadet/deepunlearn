import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from munl.datasets import get_discernible_retain_and_forget_loaders
from munl.models import get_optimizer_scheduler_criterion
from munl.unlearning.common import BaseUnlearner, save_checkpoint


# From the https://github.com/vikram2000b/bad-teaching-unlearning/blob/961d9656d869c6c18f2b1ec8f6643eed21603d82/unlearn.py#L67
def UnlearnerLoss(
    output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature
):
    labels = torch.unsqueeze(labels, dim=1)

    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

    # label 1 means forget sample
    # label 0 means retain sample
    overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out)


def unlearning_step(
    model,
    unlearning_teacher,
    full_trained_teacher,
    unlearn_data_loader,
    optimizer,
    device,
    KL_temperature,
):
    losses = []
    for batch in unlearn_data_loader:
        # We modify here as our implementation of the data loader return (data, target, origin)
        # And the implementation of the unlearning teacher uses (data, origin)
        x, _, y = batch
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)
        output = model(x)
        optimizer.zero_grad()
        loss = UnlearnerLoss(
            output=output,
            labels=y,
            full_teacher_logits=full_teacher_logits,
            unlearn_teacher_logits=unlearn_teacher_logits,
            KL_temperature=KL_temperature,
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)


def blindspot_unlearner(
    model,
    unlearning_teacher,
    full_trained_teacher,
    retain_data,
    forget_data,
    optimizer,
    epochs=10,
    device="cuda",
    KL_temperature=1,
):
    # creating the unlearning dataset.
    # NOTE: We modify using our CombinedDataset instead
    unlearning_loader = get_discernible_retain_and_forget_loaders(
        retain_data, forget_data, shuffle=True
    )

    unlearning_teacher.eval()
    full_trained_teacher.eval()
    for epoch in tqdm(range(epochs)):
        loss = unlearning_step(
            model=model,
            unlearning_teacher=unlearning_teacher,
            full_trained_teacher=full_trained_teacher,
            unlearn_data_loader=unlearning_loader,
            optimizer=optimizer,
            device=device,
            KL_temperature=KL_temperature,
        )
        print("Epoch {} Unlearning Loss {}".format(epoch + 1, loss))
        payload = {
            "unlearning_loss": loss,
        }
        save_checkpoint(
            model,
            optimizer,
            scheduler=None,
            epoch=epoch,
            payload=payload,
            filename=f"epoch_{epoch}.pth",
        )


import munl.settings


class BadTeacher(BaseUnlearner):
    ORIGINAL_LR = 0.01
    ORIGINAL_WEIGHT_DECAY = 0
    ORIGINAL_NUM_EPOCHS = 10
    ORIGINAL_KL_TEMPERATURE = 1
    ORIGINAL_BATCH_SIZE = 256

    HYPER_PARAMETERS = {
        "unlearner.cfg.kl_temperature": munl.settings.HP_TEMPERATURE,
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
        model: nn.Module,
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        val_loader: DataLoader,
    ) -> nn.Module:
        device = self.device
        (
            optimizer,
            _,
            _,
        ) = get_optimizer_scheduler_criterion(model, self.cfg)
        student = model
        teacher = copy.deepcopy(model)
        if hasattr(model, "fc"):
            num_classes = model.fc.weight.shape[0]
        else:
            num_classes = model.head.weight.shape[0]
        noisy_teacher = torchvision.models.resnet18(
            weights=None, num_classes=num_classes
        )
        print("[WARNING] We create the noisy teacher manually with 10 classes")
        student.to(device)
        teacher.to(device)
        noisy_teacher.to(device)
        blindspot_unlearner(
            model=student,
            unlearning_teacher=noisy_teacher,
            full_trained_teacher=teacher,
            retain_data=retain_loader,
            forget_data=forget_loader,
            epochs=self.cfg.num_epochs,
            optimizer=optimizer,
            device=device,
            KL_temperature=self.cfg.kl_temperature,
        )
        return student


def bad_teacher_default_optimizer():
    return {
        "type": "torch.optim.Adam",
        "learning_rate": BadTeacher.ORIGINAL_LR,
        "momentum": None,
        "weight_decay": BadTeacher.ORIGINAL_WEIGHT_DECAY,
    }


import typing as typ
from dataclasses import dataclass, field

from munl.settings import DEFAULT_MODEL_INIT_DIR, default_loaders
from munl.utils import DictConfig


@dataclass
class DefaultBadTeacherConfig:
    num_epochs: int = BadTeacher.ORIGINAL_NUM_EPOCHS
    batch_size: int = BadTeacher.ORIGINAL_BATCH_SIZE
    kl_temperature: float = BadTeacher.ORIGINAL_KL_TEMPERATURE

    optimizer: typ.Dict[str, typ.Any] = field(
        default_factory=bad_teacher_default_optimizer
    )
    scheduler: typ.Union[typ.Dict[str, typ.Any], None] = None
    criterion: typ.Union[typ.Dict[str, typ.Any], None] = None
    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)
