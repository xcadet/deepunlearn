# Adapted from https://github.com/OPTML-Group/Unlearn-Saliency/blob/master/Classification/unlearn/Wfisher.py
import typing as typ
from dataclasses import dataclass, field

import torch
from torch.autograd import grad
from torch.nn import Module
from torch.utils.data import DataLoader

from munl.settings import DEFAULT_MODEL_INIT_DIR, default_loaders
from munl.unlearning.common import BaseUnlearner
from munl.utils import DictConfig


def get_require_grad_params(model: torch.nn.Module, named=False):
    if named:
        return [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad and param.grad is not None
        ]
    else:
        return [param for param in model.parameters() if param.requires_grad]


def sam_grad(model, loss):
    params = []

    for param in get_require_grad_params(model, named=False):
        params.append(param)

    sample_grad = grad(loss, params, allow_unused=True)
    sample_grad = [x.view(-1) for x in sample_grad if x is not None]

    return torch.cat(sample_grad)


def apply_perturb(model, v, mask=None):
    curr = 0

    if mask:
        for name, param in get_require_grad_params(model, named=True):
            length = param.view(-1).shape[0]
            param.view(-1).data += v[curr : curr + length].data * mask[name].view(-1)
            curr += length

    else:
        for param in get_require_grad_params(model, named=False):
            length = param.view(-1).shape[0]
            param.view(-1).data += v[curr : curr + length].data
            curr += length


def woodfisher(model: Module, train_data_loader: DataLoader, device, criterion, v):
    model.eval()
    k_vec = torch.clone(v)
    N = 1000 
    o_vec = None
    for idx, (data, label) in enumerate(train_data_loader):
        model.zero_grad()
        data = data.to(device)
        label = label.to(device)
        output = model(data)

        loss = criterion(output, label)
        sample_grad = sam_grad(model, loss)
        with torch.no_grad():
            if o_vec is None:
                o_vec = torch.clone(sample_grad)
            else:
                tmp = torch.dot(o_vec, sample_grad)
                k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
                o_vec -= (tmp / (N + tmp)) * o_vec
        if idx > N:
            return k_vec
    return k_vec


def Wfisher(model, retain_loader, forget_loader, criterion, alpha, batch_size, device):
    retain_grad_loader = torch.utils.data.DataLoader(
        retain_loader.dataset, batch_size=batch_size, shuffle=False
    )
    retain_loader = torch.utils.data.DataLoader(
        retain_loader.dataset, batch_size=1, shuffle=False
    )
    forget_loader = torch.utils.data.DataLoader(
        forget_loader.dataset, batch_size=batch_size, shuffle=False
    )
    params = []

    for param in get_require_grad_params(model, named=False):
        params.append(param.view(-1))

    forget_grad = torch.zeros_like(torch.cat(params)).to(device)
    retain_grad = torch.zeros_like(torch.cat(params)).to(device)

    total = 0
    model.eval()
    for _, (data, label) in enumerate(forget_loader):
        model.zero_grad()
        real_num = data.shape[0]
        data = data.to(device)
        label = label.to(device)
        output = model(data)

        loss = criterion(output, label)
        f_grad = sam_grad(model, loss) * real_num
        forget_grad += f_grad
        total += real_num

    total_2 = 0
    for _, (data, label) in enumerate(retain_grad_loader):
        model.zero_grad()
        real_num = data.shape[0]
        data = data.to(device)
        label = label.to(device)
        output = model(data)

        loss = criterion(output, label)
        r_grad = sam_grad(model, loss) * real_num
        retain_grad += r_grad
        total_2 += real_num

    retain_grad *= total / ((total + total_2) * total_2)
    forget_grad /= total + total_2

    perturb = woodfisher(
        model,
        retain_loader,
        device=device,
        criterion=criterion,
        v=forget_grad - retain_grad,
    )
    apply_perturb(model, alpha * perturb)

    return model


import munl.settings


class InfluenceUnlearning(BaseUnlearner):

    # Generic Hyper
    ORIGINAL_NUM_EPOCHS = 10
    ORIGINAL_BATCH_SIZE = 256
    ORIGINAL_ALPHA = 0.2

    HYPER_PARAMETERS = {
        "unlearner.cfg.num_epochs": munl.settings.HP_NUM_EPOCHS,
        "unlearner.cfg.batch_size": munl.settings.HP_BATCH_SIZE,
        "unlearner.cfg.alpha": munl.settings.HP_NORMAL_SIGMA,
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
        alpha = self.cfg.alpha
        batch_size = self.cfg.batch_size

        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        model = Wfisher(
            model=model,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            criterion=criterion,
            alpha=alpha,
            batch_size=batch_size,
            device=device,
        )

        return model


@dataclass
class DefaultInfluenceUnlearningConfig:
    num_epochs: int = InfluenceUnlearning.ORIGINAL_NUM_EPOCHS
    batch_size: int = InfluenceUnlearning.ORIGINAL_BATCH_SIZE
    alpha: float = InfluenceUnlearning.ORIGINAL_ALPHA

    scheduler: typ.Union[typ.Dict[str, typ.Any], None] = None
    criterion: typ.Union[typ.Dict[str, typ.Any], None] = None
    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)
