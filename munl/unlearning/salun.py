# Adapted from: https://github.com/OPTML-Group/Unlearn-Sparse/blob/public/unlearn/GA.py
import copy
import typing as typ
from dataclasses import dataclass, field

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

import munl.settings as settings
from munl.datasets import RandomRelabelDataset
from munl.models import get_optimizer_scheduler_criterion
from munl.settings import DEFAULT_DEVICE, DEFAULT_MODEL_INIT_DIR, default_loaders
from munl.unlearning.common import BaseUnlearner
from munl.utils import DictConfig, get_num_classes_from_model


# NOTE: Adapted from https://github.com/OPTML-Group/Unlearn-Saliency/blob/f31f82e32429fabf39612fdf1d7cd943bfaa715e/Classification/generate_mask.py
# 1. Instead of giving the data loader we give the forget_loader directly
# 2. Swap model and loader as input
# 3. Optimizer is not given to the function
# 4. Only proceeds to 1 threshold value Which becomes an Hyper-Parameter
# 5. Return the state_dict instead of saving it
def _save_gradient_ratio(
    model: Module,
    forget_loader: DataLoader,
    criterion: Module,
    optimizer,
    threshold: float,
):
    gradients = {}

    model.eval()

    for name, param in model.named_parameters():
        gradients[name] = 0

    for i, (image, target) in enumerate(forget_loader):
        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = -criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if isinstance(gradients[name], int):
                        gradients[name] = param.grad.data.clone()
                    else:
                        gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            if isinstance(gradients[name], torch.Tensor):
                gradients[name] = torch.abs_(gradients[name])
            elif isinstance(gradients[name], int):
                continue
            else:
                raise ValueError("Unknown type of gradients")

    sorted_dict_positions = {}
    hard_dict = {}

    valid_gradients = [
        tensor.flatten()
        for tensor in gradients.values()
        if isinstance(tensor, torch.Tensor)
    ]
    all_elements = -torch.cat(valid_gradients)

    threshold_index = int(len(all_elements) * threshold)

    positions = torch.argsort(all_elements)
    ranks = torch.argsort(positions)

    start_index = 0
    for key, tensor in gradients.items():
        if isinstance(tensor, torch.Tensor):
            num_elements = tensor.numel()
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements
    return hard_dict


def save_gradient_ratio(
    model: Module,
    forget_loader: DataLoader,
    criterion: Module,
    optimizer,
    threshold: float,
):
    gradients = {}

    model.eval()

    for name, param in model.named_parameters():
        if isinstance(param.data, torch.Tensor):
            gradients[name] = torch.zeros_like(param.data)

    for i, (image, target) in enumerate(forget_loader):
        image = image.cuda()
        target = target.cuda()

        output_clean = model(image)
        loss = -criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None and isinstance(param.grad.data, torch.Tensor):
                    gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            if isinstance(gradients[name], torch.Tensor):
                gradients[name] = torch.abs_(gradients[name])

    sorted_dict_positions = {}
    hard_dict = {}

    valid_gradients = [
        tensor.flatten()
        for tensor in gradients.values()
        if isinstance(tensor, torch.Tensor)
    ]
    all_elements = -torch.cat(valid_gradients)

    threshold_index = int(len(all_elements) * threshold)

    positions = torch.argsort(all_elements)
    ranks = torch.argsort(positions)

    start_index = 0
    for key, tensor in gradients.items():
        if isinstance(tensor, torch.Tensor):
            num_elements = tensor.numel()
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements
    return hard_dict


# NOTE: Adapted from https://github.com/OPTML-Group/Unlearn-Saliency/blob/f31f82e32429fabf39612fdf1d7cd943bfaa715e/Classification/unlearn/RL.py
def RandomRelabel(
    model,
    retain_loader,
    forget_loader,
    criterion,
    optimizer,
    num_classes,
    batch_size,
    mask=None,
    device=DEFAULT_DEVICE,
):
    forget_dataset = RandomRelabelDataset(
        copy.deepcopy(forget_loader.dataset), num_classes
    )
    retain_dataset = retain_loader.dataset
    train_dataset = torch.utils.data.ConcatDataset([forget_dataset, retain_dataset])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    model.train()

    for it, (image, target) in enumerate(train_loader):
        image = image.to(device)
        target = target.to(device)
        output_clean = model(image)

        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        if mask:
            for name, param in model.named_parameters():
                if param.grad is not None and name in mask:
                    param.grad *= mask[name]

        optimizer.step()
    return model


import munl.settings


class SaliencyUnlearning(BaseUnlearner):
    ORIGINAL_MASK_LR = 0.01  
    ORIGINAL_MASK_MOMENTUM = 0.9  
    ORIGINAL_MASK_WEIGHT_DECAY = 5e-4  
    ORIGINAL_THRESHOLD = 0.5  

    ORIGINAL_LR = 0.01  
    ORIGINAL_MOMENTUM = 0  
    ORIGINAL_WEIGHT_DECAY = 0 
    ORIGINAL_NUM_EPOCHS = 10  
    ORIGINAL_BATCH_SIZE = 256  

    HYPER_PARAMETERS = {
        **settings.HYPER_PARAMETERS,
        "unlearner.cfg.threshold": munl.settings.HP_FLOAT,
        "unlearner.cfg.mask_optimizer.learning_rate": munl.settings.HP_LEARNING_RATE,
        "unlearner.cfg.mask_optimizer.momentum": munl.settings.HP_MOMENTUM,
        "unlearner.cfg.mask_optimizer.weight_decay": munl.settings.HP_WEIGHT_DECAY,
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
        # Variables preparation
        device = self.device
        batch_size = self.cfg.batch_size
        treshold = self.cfg.threshold
        num_classes = get_num_classes_from_model(model)
        num_epochs = self.cfg.num_epochs

        model.to(device)
        # Optimizer, Scheduler and Criterion
        (
            optimizer,
            _,
            _,
        ) = get_optimizer_scheduler_criterion(model, self.cfg)
        criterion = torch.nn.CrossEntropyLoss()

        # Beginning of the method
        # Therefore there are two blocks
        model_to_get_mask_from = copy.deepcopy(model)
        # 1. Obtaining the mask: Gradient Based
        mask = save_gradient_ratio(
            model_to_get_mask_from, forget_loader, criterion, optimizer, treshold
        )

        # 2. Repeated Random Labeling Training
        for _ in range(num_epochs):
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
        # End of the method
        return model


def saliency_unlearning_default_optimizer():
    return {
        "type": "torch.optim.SGD",
        "learning_rate": SaliencyUnlearning.ORIGINAL_LR,
        "momentum": SaliencyUnlearning.ORIGINAL_MOMENTUM,
        "weight_decay": SaliencyUnlearning.ORIGINAL_WEIGHT_DECAY,
    }


def saliency_unlearning_mask_optimizer():
    return {
        "type": "torch.optim.SGD",
        "learning_rate": SaliencyUnlearning.ORIGINAL_MASK_LR,
        "momentum": SaliencyUnlearning.ORIGINAL_MASK_MOMENTUM,
        "weight_decay": SaliencyUnlearning.ORIGINAL_MASK_WEIGHT_DECAY,
    }


@dataclass
class DefaultSaliencyUnlearningConfig:
    num_epochs: int = SaliencyUnlearning.ORIGINAL_NUM_EPOCHS
    batch_size: int = SaliencyUnlearning.ORIGINAL_BATCH_SIZE
    threshold: float = SaliencyUnlearning.ORIGINAL_THRESHOLD

    optimizer: typ.Dict[str, typ.Any] = field(
        default_factory=saliency_unlearning_default_optimizer
    )
    mask_optimizer: typ.Dict[str, typ.Any] = field(
        default_factory=saliency_unlearning_mask_optimizer
    )
    scheduler: typ.Union[typ.Dict[str, typ.Any], None] = None
    criterion: typ.Union[typ.Dict[str, typ.Any], None] = None
    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)
