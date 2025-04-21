import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

import munl.settings
from munl.unlearning.common import BaseUnlearner


class KGLTop7(BaseUnlearner):
    ORIGINAL_NUM_EPOCHS = 9
    ORIGINAL_FINE_EPOCHS = 4
    ORIGINAL_RESET_EPOCHS = 1
    ORIGINAL_FINE_LR = [
        0.0002,
        0.0002,
        0.0002,
        0.001,
        0.001,
        0.0001,
    ]  
    ORIGINAL_LR = 0.0002
    ORIGINAL_MOMENTUM = 0.9
    ORIGINAL_WEIGHT_DECAY = 5e-4

    ORIGINAL_ADD_ALREADY = False
    ORIGINAL_DUPLICATE_SAMPLE = True

    ORIGINAL_FINE_MOMENTUM = 0.9
    ORIGINAL_FINE_WEIGHT_DECAY = 5e-4

    ORIGINAL_NOISE_RATIO = 0.7
    ORIGINAL_NOISE = 0.04
    ORIGINAL_BATCH_SIZE = 64

    HYPER_PARAMETERS = {
        "unlearner.cfg.fine_num_epochs": munl.settings.HP_NUM_EPOCHS,
        "unlearner.cfg.fine_optimizer.learning_rate": munl.settings.HP_LEARNING_RATE,
        "unlearner.cfg.fine_optimizer.momentum": munl.settings.HP_MOMENTUM,
        "unlearner.cfg.fine_optimizer.weight_decay": munl.settings.HP_WEIGHT_DECAY,
        "unlearner.cfg.reset_num_epochs": munl.settings.HP_NUM_EPOCHS,
        "unlearner.cfg.noise_ratio": munl.settings.HP_FLOAT,
        "unlearner.cfg.origin_noise": munl.settings.HP_NORMAL_SIGMA,
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
        # The following values are copied from the notebook
        # self.epochs = self.ORIGINAL_EPOCHS
        # self.fine_epochs = self.ORIGINAL_FINE_EPOCH
        # self.lr = self.ORIGINAL_LR
        # self.fine_lr = self.ORIGINAL_FINE_LR

    # Hyper parameters were copied as provided in the notebook
    def unlearn(
        self,
        net,
        retain_loader,
        forget_loader,
        val_loader,
    ):
        device = self.device
        net.to(device)
        epochs = self.cfg.num_epochs
        fine_epoch = self.cfg.fine_num_epochs
        fine_lr = self.cfg.fine_optimizer.learning_rate
        fine_momentum = self.cfg.fine_optimizer.momentum
        fine_weight_decay = self.cfg.fine_optimizer.weight_decay
        lr = self.cfg.optimizer.learning_rate
        momentum = self.cfg.optimizer.momentum
        weight_decay = self.cfg.optimizer.weight_decay

        add_already = False
        duplicate_sample = True

        layer_select = 9
        reset_fc = True
        reset_epoch = self.cfg.reset_num_epochs

        noise_fix = True

        noise_ratio = self.cfg.noise_ratio

        noise_mode = "add"
        origin_noise = self.cfg.origin_noise  # 0.04

        criterion = nn.CrossEntropyLoss()  # The provided code did not use class weights

        def get_optimze_group(optim_keys, params, ep, epochs):
            for key in params:
                if key in optim_keys:
                    params[key].requires_grad = True
                else:
                    params[key].requires_grad = False

            optim_params = [params[key] for key in params]

            if ep >= epochs - fine_epoch:
                if hasattr(fine_lr, "__getitem__"):
                    fine_learning_rate = fine_lr[ep - epochs + fine_epoch]
                else:
                    fine_learning_rate = fine_lr
                optimizer = optim.SGD(
                    optim_params,
                    lr=fine_learning_rate,
                    momentum=fine_momentum,
                    weight_decay=fine_weight_decay,
                )
                return optimizer

            optimizer = optim.SGD(
                optim_params, lr=lr, momentum=momentum, weight_decay=weight_decay
            )
            return optimizer

        def noise_optimize_group(optim_keys, already_select, params, ep, epochs):
            if reset_fc and ep < reset_epoch:
                return params

            if ep >= epochs - fine_epoch:
                return params

            if noise_fix:
                optim_keys = np.random.choice(
                    [key for key in optim_keys if "bn" not in key],
                    int(noise_ratio * layer_select),
                    replace=False,
                )
                optim_keys = sorted(optim_keys)

            for key in optim_keys:
                if "bn" in key:
                    continue

                if np.random.rand() > noise_ratio and not noise_fix:
                    print(f"× {key}")
                    continue
                print(f"√ {key}")

                noise = torch.randn_like(params[key].data)

                if key in already_select:
                    continue

                assert noise_mode == "add", "The reported method used the add mode"
                if noise_mode == "add":
                    params[key].data = params[key].data + noise * origin_noise
            return params

        def select_optim_keys(params_keys, already_select, ep, epochs, add_already):
            if ep >= epochs - fine_epoch:
                return params_keys, already_select

            must_to_op = [key for key in params_keys if "bn" in key]
            select_to_op = [key for key in params_keys if "bn" not in key]
            if not duplicate_sample:
                select_to_op = [
                    key for key in select_to_op if key not in already_select
                ]

            if len(select_to_op) == 0:
                select_to_op = already_select
                # clear the already_select
                for i in range(len(already_select)):
                    already_select.pop()

            keys_ = np.random.choice(
                select_to_op, min(layer_select, len(select_to_op)), replace=False
            ).tolist()

            # already_select = already_select + keys_
            keys = keys_ + must_to_op

            if add_already:
                keys = keys + already_select

            keys = list(set(keys))
            keys = sorted(keys)

            return keys, already_select

        params = net.named_parameters()
        params = dict(params)
        params_keys = list(params.keys())
        for key in params_keys:
            print(f"{key: <30}: {params[key].requires_grad}")

        print(f"there are {len(params_keys)} params to optimize.")

        already_select = list()

        if reset_fc:
            print("reset fc ------")
            if hasattr(
                net, "fc"
            ):  
                net.fc.reset_parameters()  # NOTE: This is hardcoded and required a manual update
            elif hasattr(net, "head"):
                net.head.reset_parameters()  # NOTE: This is hardcoded and required a manual update
            else:
                raise NotImplementedError(
                    "The model does not have a fc or head attribute"
                )

        for ep in range(epochs):
            net.train()
            print(f"Tranning....{ep}")

            optim_keys, already_select = select_optim_keys(
                params_keys, already_select, ep, epochs, add_already
            )
            params = noise_optimize_group(
                optim_keys, already_select, params, ep, epochs
            )
            optimizer = get_optimze_group(optim_keys, params, ep, epochs)
            already_select = list(set(already_select + optim_keys))
            # ----- print ---- #
            for k in optim_keys:
                if params[k].requires_grad and "bn" not in k:
                    print("- ", k)

            for sample in retain_loader:
                inputs, targets = sample
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = net(inputs)
                # smoothed_targets = label_smooth(targets, 10, 0.1)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            # scheduler.step()

            # compute the val_loader acc
            net.eval()
            total_correct = 0
            total_samples = 0

            with torch.no_grad():
                for sample in val_loader:
                    inputs, targets = sample
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = net(inputs)
                    _, predicted = torch.max(outputs, dim=1)

                    total_correct += (predicted == targets).sum().item()
                    total_samples += targets.size(0)

            accuracy = total_correct / total_samples
            print(f"Validation Accuracy: {accuracy:.2%}")

        net.eval()
        return net


def kgl7_default_optimizer():
    return {
        "type": "torch.optim.SGD",
        "learning_rate": KGLTop7.ORIGINAL_LR,
        "momentum": KGLTop7.ORIGINAL_MOMENTUM,
        "weight_decay": KGLTop7.ORIGINAL_WEIGHT_DECAY,
    }


def kgl7_default_fine_optimizer():
    return {
        "type": "torch.optim.SGD",
        "learning_rate": KGLTop7.ORIGINAL_FINE_LR,
        "momentum": KGLTop7.ORIGINAL_FINE_MOMENTUM,
        "weight_decay": KGLTop7.ORIGINAL_FINE_WEIGHT_DECAY,
    }


import typing as typ
from dataclasses import dataclass, field

from munl.settings import DEFAULT_MODEL_INIT_DIR, default_loaders
from munl.utils import DictConfig


@dataclass
class DefaultKGLTop7Config:
    num_epochs: int = KGLTop7.ORIGINAL_NUM_EPOCHS
    batch_size: int = KGLTop7.ORIGINAL_BATCH_SIZE
    fine_num_epochs: int = KGLTop7.ORIGINAL_FINE_EPOCHS

    optimizer: typ.Dict[str, typ.Any] = field(default_factory=kgl7_default_optimizer)
    fine_optimizer: typ.Dict[str, typ.Any] = field(
        default_factory=kgl7_default_fine_optimizer
    )

    reset_num_epochs: int = KGLTop7.ORIGINAL_RESET_EPOCHS
    noise_ratio: float = KGLTop7.ORIGINAL_NOISE_RATIO
    origin_noise: float = KGLTop7.ORIGINAL_NOISE

    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)
