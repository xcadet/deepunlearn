import copy
import typing as typ
from dataclasses import dataclass, field
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
from omegaconf import DictConfig

import munl.settings
from munl.settings import DEFAULT_MODEL_INIT_DIR, default_loaders
from munl.unlearning.common import BaseUnlearner
from munl.utils import DictConfig


def kl_loss_fn(outputs, dist_target):
    kl_loss = F.kl_div(
        torch.log_softmax(outputs, dim=1),
        dist_target,
        log_target=True,
        reduction="batchmean",
    )
    return kl_loss


def entropy_loss_fn(outputs, labels, dist_target, class_weights):
    ce_loss = F.cross_entropy(outputs, labels, weight=class_weights)
    entropy_dist_target = torch.sum(-torch.exp(dist_target) * dist_target, dim=1)
    entropy_outputs = torch.sum(
        -torch.softmax(outputs, dim=1) * torch.log_softmax(outputs, dim=1), dim=1
    )
    entropy_loss = F.mse_loss(entropy_outputs, entropy_dist_target)
    return ce_loss + entropy_loss


def get_class_weights(loader):
    # The original implementation assuemt that the class weights can be obtained directly
    # We compute the class weights based from the dataset under the loader
    all_targets = []
    for sample in loader:
        _, targets = sample
        all_targets.extend(targets.tolist())
    targets, counts = torch.unique(torch.tensor(all_targets), return_counts=True)
    ordered_targets_and_counts = sorted(
        zip(targets.tolist(), counts.tolist()), key=lambda x: x[0]
    )
    only_counts = torch.Tensor([x[1] for x in ordered_targets_and_counts]).to(
        torch.float
    )
    class_weights = only_counts**-0.1
    return class_weights


def prune_model(net, amount=0.95, rand_init=True):
    # Modules to prune
    modules = list()
    for k, m in enumerate(net.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            modules.append((m, "weight"))
            if m.bias is not None:
                modules.append((m, "bias"))

    # Prune criteria
    prune.global_unstructured(
        modules,
        # pruning_method=prune.RandomUnstructured,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Perform the prune
    for k, m in enumerate(net.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            prune.remove(m, "weight")
            if m.bias is not None:
                prune.remove(m, "bias")

    # Random initialization
    if rand_init:
        for k, m in enumerate(net.modules()):
            if isinstance(m, nn.Conv2d):
                mask = m.weight == 0
                c_in = mask.shape[1]
                k = 1 / (c_in * mask.shape[2] * mask.shape[3])
                randinit = (torch.rand_like(m.weight) - 0.5) * 2 * sqrt(k)
                m.weight.data[mask] = randinit[mask]
            if isinstance(m, nn.Linear):
                mask = m.weight == 0
                c_in = mask.shape[1]
                k = 1 / c_in
                randinit = (torch.rand_like(m.weight) - 0.5) * 2 * sqrt(k)
                m.weight.data[mask] = randinit[mask]


# PRMQ - Prune, Retrain, Mask, Quantize


class KGLTop4(BaseUnlearner):
    ORIGINAL_NUM_EPOCHS = 3.2
    ORIGINAL_BATCH_SIZE = 64
    ORIGINAL_LR = 0.0005

    ORIGINAL_MOMENTUM = 0.9
    ORIGINAL_WEIGHT_DECAY = 5e-4
    ORIGINAL_PRUNE_AMOUNT = 0.99

    HYPER_PARAMETERS = {
        "unlearner.cfg.prune_amount": munl.settings.HP_FLOAT,
        **munl.settings.HYPER_PARAMETERS,
        "unlearner.cfg.num_epochs": munl.settings.HP_NUM_EPOCHS_FLOAT,
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
        # The following values are copied from the notebook

    # Hyper parameters were copied as provided in the notebook
    def unlearn(
        self,
        net,
        retain_loader,
        forget_loader,
        val_loader,
    ):
        lr = self.cfg.optimizer.learning_rate
        momentum = self.cfg.optimizer.momentum
        weight_decay = self.cfg.optimizer.weight_decay
        device = self.device
        net.to(device)
        epochs = self.cfg.num_epochs
        prune_amount = self.cfg.prune_amount

        max_iters = int(len(retain_loader) * epochs)
        # (
        # optimizer,
        # scheduler,
        # criterion,
        # ) = get_optimizer_scheduler_criterion(net, self.cfg)
        print(
            "Optimizer type is pre-set by the notebook, same with scheduler and criterion."
        )
        optimizer = optim.SGD(
            net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        initial_net = copy.deepcopy(net)

        class_weights = get_class_weights(retain_loader).to(device)
        net.train()
        initial_net.eval()

        num_iters = 0
        running = True
        prune_model(net, prune_amount, True)
        while running:
            net.train()
            for sample in retain_loader:
                inputs, targets = sample
                inputs, targets = inputs.to(device), targets.to(device)

                # Get target distribution
                with torch.no_grad():
                    original_outputs = initial_net(inputs)
                    preds = torch.log_softmax(original_outputs, dim=1)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = entropy_loss_fn(outputs, targets, preds, class_weights)
                loss.backward()
                optimizer.step()

                num_iters += 1
                # Stop at max iters
                if num_iters > max_iters:
                    running = False
                    break

        net.eval()
        net = net.to(torch.half)
        net = net.to(torch.float)
        return net


def kgl4_default_optimizer():
    return {
        "type": "torch.optim.SGD",
        "learning_rate": KGLTop4.ORIGINAL_LR,
        "momentum": KGLTop4.ORIGINAL_MOMENTUM,
        "weight_decay": KGLTop4.ORIGINAL_WEIGHT_DECAY,
    }


#
@dataclass
class DefaultKGLTop4Config:
    num_epochs: float = KGLTop4.ORIGINAL_NUM_EPOCHS
    batch_size: int = KGLTop4.ORIGINAL_BATCH_SIZE
    prune_amount: float = KGLTop4.ORIGINAL_PRUNE_AMOUNT
    optimizer: typ.Dict[str, typ.Any] = field(default_factory=kgl4_default_optimizer)
    scheduler: typ.Union[typ.Dict[str, typ.Any], None] = None
    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)
