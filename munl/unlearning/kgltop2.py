import typing as typ
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Subset

import munl.settings
from munl.settings import DEFAULT_MODEL_INIT_DIR, default_loaders
from munl.unlearning.common import BaseUnlearner


class Masker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(mask)
        return x

    @staticmethod
    def backward(ctx, grad):
        (mask,) = ctx.saved_tensors
        return grad * mask, None


class MaskConv2d(nn.Conv2d):
    def __init__(
        self,
        mask,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device="cpu",
    ):
        super(MaskConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device=device,
        )
        self.mask = mask

    def forward(self, input):
        masked_weight = Masker.apply(self.weight, self.mask)
        return super(MaskConv2d, self)._conv_forward(input, masked_weight, self.bias)


def set_layer(model, layer_name, layer):
    def set_nested_attr(obj, attr, value):
        pre, _, post = attr.rpartition(".")
        if pre:
            target = get_nested_attr(obj, pre)
            if post.isdigit():
                target[int(post)] = value
            else:
                setattr(target, post, value)
        else:
            setattr(obj, post, value)

    def get_nested_attr(obj, attr):
        for part in attr.split("."):
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj

    set_nested_attr(model, layer_name, layer)

@torch.no_grad()
def replace_maskconv(model, device):
    print("Remove Maskconv")
    for name, m in list(model.named_modules()):
        if isinstance(m, MaskConv2d):
            conv = nn.Conv2d(
                m.in_channels,
                m.out_channels,
                m.kernel_size,
                m.stride,
                m.padding,
                m.dilation,
                m.groups,
                m.bias != None,
                m.padding_mode,
                device=device,
            )
            conv.weight.data = m.weight
            conv.bias = m.bias
            set_layer(model, name, conv)


@torch.no_grad()
def re_init_model_snip_ver2_little_grad(
    model, px, device
):
    print(
        "Apply Unstructured re_init_model_snip_ver2_little_grad Globally (all conv layers)"
    )
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Conv2d):
            mask = torch.zeros_like(m.weight, device=device).bool()
            nparams_toprune = round(px * mask.nelement())

            out_c, in_c, ke, _ = mask.shape
            value = -m.weight.grad.abs()
            topk = torch.topk(value.view(-1), k=nparams_toprune)
            mask.view(-1)[topk.indices] = True
            grad_mask = mask.clone().float()
            grad_mask[grad_mask == 0] += 0.1

            new_conv = MaskConv2d(
                grad_mask,
                m.in_channels,
                m.out_channels,
                m.kernel_size,
                m.stride,
                m.padding,
                m.dilation,
                m.groups,
                m.bias != None,
                m.padding_mode,
                device=device,
            )
            nn.init.kaiming_normal_(
                new_conv.weight, mode="fan_out", nonlinearity="relu"
            )

            new_conv.weight.data[~mask] = m.weight[~mask]

            set_layer(model, name, new_conv)


def get_grads_for_snip(model, retain_loader, forget_loader, device):
    indices = torch.randperm(
        len(retain_loader.dataset), dtype=torch.int32, device=device
    )[: len(forget_loader.dataset)]
    retain_dataset = Subset(retain_loader.dataset, indices)
    retain_loader = DataLoader(retain_dataset, batch_size=64, shuffle=True)

    model.zero_grad()
    for sample in retain_loader:
        inputs, targets = sample
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()

    for sample in forget_loader:
        inputs, targets = sample
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = -F.cross_entropy(outputs, targets)
        loss.backward()


class LinearAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, num_annealing_steps, num_total_steps):
        self.num_annealing_steps = num_annealing_steps
        self.num_total_steps = num_total_steps

        super().__init__(optimizer)

    def get_lr(self):
        if self._step_count <= self.num_annealing_steps:
            return [
                base_lr * self._step_count / self.num_annealing_steps
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                * (self.num_total_steps - self._step_count)
                / (self.num_total_steps - self.num_annealing_steps)
                for base_lr in self.base_lrs
            ]


class KGLTop2(BaseUnlearner):
    ORIGINAL_BATCH_SIZE = 64
    ORIGINAL_INIT_RATE = 0.3
    ORIGINAL_LR = 0.001
    ORIGINAL_NUM_EPOCHS = 5
    ORIGINAL_WEIGHT_DECAY = 5e-4
    ORIGINAL_MOMENTUM = 0.9

    HYPER_PARAMETERS = {
        "unlearner.cfg.init_rate": munl.settings.HP_FLOAT,
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
        net,
        retain_loader,
        forget_loader,
        val_loader,
    ):
        init_rate = self.cfg.init_rate
        lr = self.cfg.optimizer.learning_rate
        weight_decay = self.cfg.optimizer.weight_decay
        momentum = self.cfg.optimizer.momentum
        epochs = self.cfg.num_epochs
        device = self.device
        net.to(device)
        replace_maskconv(net, device=device)
        get_grads_for_snip(net, retain_loader, forget_loader, device=device)
        re_init_model_snip_ver2_little_grad(net, init_rate, device=device)

        """Simple unlearning by finetuning."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )

        scheduler = LinearAnnealingLR(
            optimizer,
            num_annealing_steps=(epochs + 1) // 2,
            num_total_steps=epochs + 1,
        )

        net.train()

        for _ in range(epochs):
            for sample in retain_loader:
                inputs, targets = sample
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

            scheduler.step()
        net.eval()
        return net


def kgl2_default_optimizer():
    return {
        "learning_rate": KGLTop2.ORIGINAL_LR,
        "momentum": KGLTop2.ORIGINAL_MOMENTUM,
        "weight_decay": KGLTop2.ORIGINAL_WEIGHT_DECAY,
    }


from munl.utils import DictConfig


@dataclass
class DefaultKGLTop2Config:
    num_epochs: int = KGLTop2.ORIGINAL_NUM_EPOCHS
    batch_size: int = KGLTop2.ORIGINAL_BATCH_SIZE
    init_rate: float = KGLTop2.ORIGINAL_INIT_RATE
    optimizer: typ.Dict[str, typ.Any] = field(default_factory=kgl2_default_optimizer)
    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)
