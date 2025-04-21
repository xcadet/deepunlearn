import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torchvision  # type:ignore

from omegaconf import DictConfig
from torch.nn import Module
from torch.optim import Optimizer

import thirdparty.tiny_vit as tiny_vit
from munl import DEFAULT_DEVICE
from munl.models.vit11m import create_tiny_vit_with_num_classes_and_size



logger = logging.getLogger(__name__)


def format_model_path(
    output_dir: str,
    num_classes: int,
    model_name: str,
    seed: int,
    img_size: int,
    split_ndx: Optional[int] = None,
    forget_ndx: Optional[int] = None,
) -> str:
    """Construct the model path based on the output directory, number of classes,
    model name, and seed.

    Args:
        output_dir (str): Directory where the model should be
        num_classes (int): Number of classes in the output layer
        model_name (str): Name of the architecture
        seed (int): Initialization Seed of the model
        img_size (int): Size of the input

    Returns:
        str: Path to the model
    """
    assert (split_ndx is None and forget_ndx is None) or (
        split_ndx is not None and forget_ndx is not None
    ), "Not implemented"
    path = None
    if model_name == "resnet18":
        path = os.path.join(output_dir, f"{num_classes}_{model_name}_{seed}.pth")
    else:
        path = os.path.join(
            output_dir, f"{num_classes}_{img_size}{model_name}_{seed}.pth"
        )
    if split_ndx is not None and forget_ndx is not None:
        path = path.replace(".pth", f"_{split_ndx}_{forget_ndx}.pth")
    assert path is not None
    return path


def get_model(model_name: str, num_classes: int, img_size: int) -> Module:
    if model_name == "vit11m":
        print(
            "Creating a VIT modle with num_classes:",
            num_classes,
            "and img_size:",
            img_size,
        )
    supported_models = {
        "resnet18": torchvision.models.resnet18(num_classes=num_classes),
        "vit11m": create_tiny_vit_with_num_classes_and_size(
            num_classes=num_classes, img_size=img_size
        ),
    }
    assert (
        model_name in supported_models
    ), f"Model {model_name} not supported. Supported models: {supported_models}"
    model = supported_models[model_name]
    return model


def load_model_state_dict(
    model: torch.nn.Module,
    output_dir: str,
    num_classes: int,
    model_name: str,
    seed: int,
    img_size: int,
    split_ndx: Optional[int] = None,
    forget_ndx: Optional[int] = None,
):
    print(f"Trying to load state dict with {split_ndx, forget_ndx}")
    input_model_path = format_model_path(
        output_dir,
        num_classes,
        model_name,
        seed,
        img_size,
        split_ndx=split_ndx,
        forget_ndx=forget_ndx,
    )
    assert os.path.exists(
        input_model_path
    ), f"Model {model_name} not found at {input_model_path}"
    logger.info(f"Loading model {model_name} from '{input_model_path}'")
    state_dict = torch.load(input_model_path, map_location=torch.device(DEFAULT_DEVICE))
    model.load_state_dict(state_dict)
    return model


def save_model_state_dict(
    model: torch.nn.Module,
    output_dir: str,
    num_classes: int,
    model_name: str,
    seed: int,
    img_size: int,
):
    output_model_path = format_model_path(
        output_dir, num_classes, model_name, seed, img_size=img_size
    )
    logger.info(f"Saving model {model_name} to {output_model_path}")
    dirname = os.path.dirname(output_model_path)
    os.makedirs(dirname, exist_ok=True)
    torch.save(model.state_dict(), output_model_path)


def get_optimizer_from_cfg(model: nn.Module, cfg: DictConfig) -> Optimizer:
    assert isinstance(model, nn.Module)
    assert isinstance(cfg.optimizer, DictConfig)

    optimizer: Optimizer
    if cfg.optimizer.type == "torch.optim.SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.learning_rate,
            momentum=cfg.optimizer.momentum,
            weight_decay=cfg.optimizer.weight_decay,
        )
    elif cfg.optimizer.type == "torch.optim.Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.learning_rate,
            weight_decay=cfg.optimizer.weight_decay,
        )
    else:
        raise NotImplementedError()
    assert isinstance(optimizer, Optimizer)
    return optimizer


def get_scheduler_from_cfg(
    optimizer: Optimizer, unlearner_cfg: DictConfig
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    if unlearner_cfg.scheduler is None:
        return None
    if unlearner_cfg.scheduler.type == "torch.optim.lr_scheduler.CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=unlearner_cfg.num_epochs
        )
    else:
        raise NotImplementedError()
    return scheduler


def get_criterion_from_cfg(cfg: DictConfig) -> Optional[Module]:
    criterion: Optional[Module]
    if cfg.criterion is None:
        return None
    if cfg.criterion.type == "torch.nn.CrossEntropyLoss":
        criterion = torch.nn.CrossEntropyLoss()
    elif cfg.criterion.type == "torch.nn.MSELoss":
        criterion = torch.nn.MSELoss()
    else:
        raise NotImplementedError()
    return criterion


def get_optimizer_scheduler_criterion(model, cfg: DictConfig):
    optimizer = get_optimizer_from_cfg(model, cfg)
    scheduler = get_scheduler_from_cfg(optimizer, cfg)
    criterion = get_criterion_from_cfg(cfg)
    return optimizer, scheduler, criterion


def get_loaded_model(
    model_name: str,
    num_classes: int,
    weights_path: Path,
    model_seed: int,
    img_size: int,
    split_ndx: Optional[int] = None,
    forget_ndx: Optional[int] = None,
) -> nn.Module:
    """Load model based on the model name, num classes, and weights path.

    Args:
        model_name (str): Name of the model to load
        num_classes (int): Number of classes in the final layer
        weights_path (Path): Path where the weights are stored
        model_seed (int): Seed of the model to load from

    Returns:
        nn.Module: Model with loaded weights
    """
    model = get_model(model_name=model_name, num_classes=num_classes, img_size=img_size)
    loaded_model = load_model_state_dict(
        model,
        str(weights_path),
        num_classes,
        model_name,
        model_seed,
        img_size=img_size,
        split_ndx=split_ndx,
        forget_ndx=forget_ndx,
    )
    return loaded_model


def get_model_from_cfg(
    root: Path,
    model_cfg: DictConfig,
    unlearner_cfg: DictConfig,
    num_classes: int,
    model_seed: int,
    img_size: int,
    split_ndx: Optional[int] = None,
    forget_ndx: Optional[int] = None,
) -> nn.Module:
    """Obtain a loaded mode based on the model and unlearner configurations.

    Args:
        root (Path): Directory to look from.
        model_cfg (DictConfig): Configuration for the model
        unlearner_cfg (DictConfig): Configuration for the Unlearner mechanism
        num_classes (int): Number of classes in the final layher
        model_seed (int): Seed to load the model from

    Returns:
        nn.Module: Model with loaded wieghts
    """

    load_path = root / unlearner_cfg.model_initializations_dir
    print("Attempting to load from load_path:", load_path)
    model = get_loaded_model(
        model_name=model_cfg.name,
        num_classes=num_classes,
        weights_path=load_path,
        model_seed=model_seed,
        img_size=img_size,
        split_ndx=split_ndx,
        forget_ndx=forget_ndx,
    )
    return model


def freeze_model(model: Module) -> Module:
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def unfreeze_modules(modules: List[Module]) -> None:
    for module in modules:
        for param in module.parameters():
            param.requires_grad_(True)


def get_model_classifier(model: Module) -> Module:
    if hasattr(model, "fc"):
        return model.fc
    elif hasattr(model, "head"):
        return model.head
    else:
        raise NotImplementedError("Unsupported model type")


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def reinitialize_model(model: Module) -> Module:
    model.apply(init_weights)
    return model


def reinialize_modules(modules: List[Module]) -> None:
    for module in modules:
        module.apply(init_weights)


def get_model_blocks(model: Module):
    if isinstance(model, torchvision.models.ResNet):
        return get_resnet_blocks(model)
    elif isinstance(model, tiny_vit.TinyViT):
        return get_vit_blocks(model)
    else:
        raise NotImplementedError()


def get_vit_blocks(model: Module) -> List[tiny_vit.BasicLayer]:
    blocks = [layer for layer in model.layers if isinstance(layer, tiny_vit.BasicLayer)]
    return blocks


def get_resnet_blocks(model: Module) -> List[Module]:
    blocks = [getattr(model, f"layer{k}") for k in range(1, 5)]
    return blocks


def select_last_k_blocks(model: Module, num_blocks: int) -> List[Module]:
    assert isinstance(model, torchvision.models.ResNet) or isinstance(
        model, tiny_vit.TinyViT
    ), "Only resnet18 and tiny_vit are supported."
    blocks = get_model_blocks(model)
    return blocks[-num_blocks:]
