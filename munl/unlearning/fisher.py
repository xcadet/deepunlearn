# Fisher Forgettinb
# Adapted from https://github.com/OPTML-Group/Unlearn-Sparse/tree/public

import copy
import typing as typ
from dataclasses import dataclass, field

import torch
from omegaconf import DictConfig
from torch.autograd import grad
from tqdm import tqdm

import munl.settings
from munl.settings import DEFAULT_MODEL_INIT_DIR, default_loaders
from munl.unlearning.common import BaseUnlearner
from munl.utils import DictConfig, get_num_classes_from_model

# Adapted from https://github.com/OPTML-Group/Unlearn-Sparse/tree/public
class FisherForgetting(BaseUnlearner):
    # https://github.com/OPTML-Group/Unlearn-Sparse/blob/80ba55d36d0871465c2d3e3896db26a98070e065/arg_parser.py#L137
    ORIGINAL_ALPHA = 0.2
    ORIGINAL_BATCH_SIZE = 64

    HYPER_PARAMETERS = {
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
        net,
        retain_loader,
        forget_loader,
        val_loader,
    ):
        device = self.device
        alpha = self.cfg.alpha
        batch_size = self.cfg.batch_size
        num_classes = get_num_classes_from_model(net)
        net.to(device)

        print(
            "Criterion, opimizer and scheduler are defined as provided by the authors."
        )
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        net.train()
        net = fisher_new(
            model=net,
            retain_loader=retain_loader,
            criterion=criterion,
            num_classes=num_classes,
            alpha=alpha,
            batch_size=batch_size,
            device=device,
        )
        net.eval()
        return net



def fisher_default_optimizer():
    return {
        "type": None,
    }


@dataclass
class DefaultFisherForgettingConfig:
    batch_size: int = FisherForgetting.ORIGINAL_BATCH_SIZE
    alpha: float = FisherForgetting.ORIGINAL_ALPHA

    optimizer: typ.Dict[str, typ.Any] = field(default_factory=fisher_default_optimizer)
    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)


def fisher_information_martix(model, train_dl, device):
    model.eval()
    fisher_approximation = []
    for parameter in model.parameters():
        fisher_approximation.append(torch.zeros_like(parameter).to(device))
    total = 0
    for i, (data, label) in enumerate(tqdm(train_dl)):
        data = data.to(device)
        label = label.to(device)
        predictions = torch.log_softmax(model(data), dim=-1)
        real_batch = data.shape[0]

        epsilon = 1e-8
        for i in range(real_batch):
            label_i = label[i]
            prediction = predictions[i][label_i]
            gradient = grad(
                prediction, model.parameters(), retain_graph=True, create_graph=False
            )
            for j, derivative in enumerate(gradient):
                fisher_approximation[j] += (derivative + epsilon) ** 2
        total += real_batch
    for i, parameter in enumerate(model.parameters()):
        fisher_approximation[i] = fisher_approximation[i] / total

    return fisher_approximation


def fisher(data_loaders, model, criterion, args):
    retain_loader = data_loaders["retain"]

    device = f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu"
    fisher_approximation = fisher_information_martix(model, retain_loader, device)
    for i, parameter in enumerate(model.parameters()):
        noise = torch.sqrt(args.alpha / fisher_approximation[i]).clamp(
            max=1e-3
        ) * torch.empty_like(parameter).normal_(0, 1)
        noise = noise * 10 if parameter.shape[-1] == 10 else noise
        print(torch.max(noise))
        parameter.data = parameter.data + noise
    return model


def hessian(dataset, model, loss_fn, batch_size: int, device):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    for p in model.parameters():
        p.grad_acc = 0
        p.grad2_acc = 0

    for data, orig_target in tqdm(train_loader):
        data, orig_target = data.to(device), orig_target.to(device)
        output = model(data)
        prob = torch.nn.functional.softmax(output, dim=-1).data

        for y in range(output.shape[1]):
            target = torch.empty_like(orig_target).fill_(y)
            loss = loss_fn(output, target)
            model.zero_grad()
            loss.backward(retain_graph=True)
            for p in model.parameters():
                if p.requires_grad:
                    p.grad2_acc += torch.mean(prob[:, y]) * p.grad.data.pow(2)

    for p in model.parameters():
        p.grad2_acc /= len(train_loader)


def get_mean_var(p, num_classes: int, alpha: float, is_base_dist=False):
    var = copy.deepcopy(1.0 / (p.grad2_acc + 1e-8))
    var = var.clamp(max=1e3)
    if p.shape[0] == num_classes:
        var = var.clamp(max=1e2)
    var = alpha * var
    if p.ndim > 1:
        var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
    if not is_base_dist:
        mu = copy.deepcopy(p.data0.clone())
    else:
        mu = copy.deepcopy(p.data0.clone())

    if p.shape[0] == num_classes:
        var *= 10
    elif p.ndim == 1:
        var *= 10
    return mu, var


def fisher_new(
    retain_loader,
    model,
    criterion,
    num_classes: int,
    alpha: float,
    batch_size: int,
    device: str,
):
    dataset = retain_loader.dataset
    for p in model.parameters():
        p.data0 = copy.deepcopy(p.data.clone())
    hessian(dataset, model, criterion, batch_size=batch_size, device=device)
    for i, p in enumerate(model.parameters()):
        mu, var = get_mean_var(p, num_classes, alpha, False)
        p.data = mu + var.sqrt() * torch.empty_like(p.data).normal_()
    return model
