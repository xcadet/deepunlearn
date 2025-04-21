# Adapted from: https://github.com/OPTML-Group/Unlearn-Sparse/blob/public/unlearn/GA.py
import copy
import typing as typ
from argparse import Namespace
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import DataLoader

import munl.settings
from munl.settings import DEFAULT_DEVICE, DEFAULT_MODEL_INIT_DIR, default_loaders
from munl.unlearning.common import BaseUnlearner
from munl.utils import DictConfig, get_num_workers_from_shuffle
from thirdparty.repdistiller.distiller_zoo import DistillKL
from thirdparty.repdistiller.helper.loops import train_distill
from thirdparty.repdistiller.helper.util import (
    adjust_learning_rate as sgda_adjust_learning_rate,
)


class SCRUBUnlearning(BaseUnlearner):
    # Specific to the original implementation
    ORIGNAL_GAMMA = 1
    ORIGINAL_ALPHA = 0.5
    ORIGINAL_BETA = 0.1  
    ORIGINAL_SMOOTHING = 0.5
    ORIGINAL_MSTEPS = 3
    ORIGINAL_CLIP = 0.2
    ORIGINAL_SSTART = 10
    ORGIGINAL_KD_T = 2
    ORIGINAL_LR_DECAY_RATE = 0.1

    # Generic Hyper
    ORIGINAL_LR = 0.0005
    ORIGINAL_MOMENTUM = 0.9
    ORIGINAL_WEIGHT_DECAY = 0.1
    ORIGINAL_SGDA_NUM_EPOCHS = 10

    ORIGINAL_BATCH_SIZE = 32  
    ORIGINAL_FORGET_BATCH_SIZE = 64


    HYPER_PARAMETERS = {
        "unlearner.cfg.sgda_num_epochs": munl.settings.HP_NUM_EPOCHS,
        "unlearner.cfg.batch_size": munl.settings.HP_BATCH_SIZE,
        "unlearner.cfg.gamma": munl.settings.HP_FLOAT,
        "unlearner.cfg.alpha": munl.settings.HP_FLOAT,
        "unlearner.cfg.beta": munl.settings.HP_FLOAT,
        "unlearner.cfg.smoothing": munl.settings.HP_FLOAT,
        "unlearner.cfg.msteps": munl.settings.HP_NUM_EPOCHS,
        "unlearner.cfg.clip": munl.settings.HP_FLOAT,
        "unlearner.cfg.sstart": munl.settings.HP_NUM_EPOCHS,
        "unlearner.cfg.kd_t": munl.settings.HP_TEMPERATURE,
        "unlearner.cfg.forget_batch_size": munl.settings.HP_BATCH_SIZE,
        "unlearner.cfg.learning_rate": munl.settings.HP_LEARNING_RATE,
        "unlearner.cfg.momentum": munl.settings.HP_MOMENTUM,
        "unlearner.cfg.weight_decay": munl.settings.HP_WEIGHT_DECAY,
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
        sstart = self.cfg.sstart
        sgda_epochs = self.cfg.sgda_num_epochs
        lr_decay_epochs = [
            int(sgda_epochs * 0.5),
            int(sgda_epochs * 0.8),
            int(sgda_epochs * 0.9),
        ]  # Based on original values
        sgda_learning_rate = self.cfg.learning_rate
        lr_decay_rate = self.cfg.lr_decay_rate
        sgda_weight_decay = self.cfg.weight_decay
        kd_t = self.cfg.kd_t
        msteps = self.cfg.msteps
        retain_batch_size = self.cfg.batch_size
        forget_batch_size = self.cfg.forget_batch_size
        shuffle = True  
        opt = Namespace
        opt.distill = "kd"
        opt.smoothing = self.cfg.smoothing
        opt.gamma = self.cfg.gamma
        opt.beta = self.cfg.beta
        opt.alpha = self.cfg.alpha

        retain_loader = DataLoader(
            retain_loader.dataset,
            batch_size=retain_batch_size,
            shuffle=shuffle,
            num_workers=get_num_workers_from_shuffle(shuffle),
        )

        forget_batch_size = DataLoader(
            forget_loader.dataset,
            batch_size=forget_batch_size,
            shuffle=shuffle,
            num_workers=get_num_workers_from_shuffle(shuffle),
        )

        # Models preparation
        model.to(device)
        teacher = copy.deepcopy(model)
        student = model

        module_list = nn.ModuleList([])
        module_list.append(student)
        trainable_list = nn.ModuleList([])
        trainable_list.append(student)

        criterion_cls = nn.CrossEntropyLoss()
        criterion_div = DistillKL(kd_t)
        criterion_kd = DistillKL(kd_t)

        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_cls)  # classification loss
        criterion_list.append(
            criterion_div
        )  # KL divergence loss, original knowledge distillation
        criterion_list.append(criterion_kd)  # other knowledge distillation loss

        swa_model = torch.optim.swa_utils.AveragedModel(student, avg_fn=avg_fn)

        optimizer = torch.optim.Adam(
            trainable_list.parameters(),
            lr=sgda_learning_rate,
            weight_decay=sgda_weight_decay,
        )

        module_list.append(teacher)
        module_list.to(device)
        criterion_list.to(device)
        swa_model.to(device)

        for epoch in range(1, sgda_epochs + 1):
            lr = sgda_adjust_learning_rate(
                epoch=epoch,
                lr_decay_epochs=lr_decay_epochs,
                sgda_learning_rate=sgda_learning_rate,
                lr_decay_rate=lr_decay_rate,
                optimizer=optimizer,
            )

            print("==> scrub unlearning ...")
            maximize_loss = 0
            if epoch <= msteps:
                maximize_loss = train_distill(
                    epoch,
                    forget_loader,
                    module_list,
                    swa_model,
                    criterion_list,
                    optimizer,
                    opt,
                    "maximize",
                )
            train_acc, train_loss = train_distill(
                epoch,
                retain_loader,
                module_list,
                swa_model,
                criterion_list,
                optimizer,
                opt,
                "minimize",
            )
            if epoch >= sstart:
                swa_model.update_parameters(student)
            print(
                "maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(
                    maximize_loss, train_loss, train_acc
                )
            )
        return student


# NOTE: Adapted from  https://github.com/meghdadk/SCRUB/blob/main/small_scale_unlearning.ipynb
# 1. num_averaged was unused, beta was assumed to be from outside the function
def avg_fn(averaged_model_parameter, model_parameter, beta):
    return (1 - beta) * averaged_model_parameter + beta * model_parameter

@dataclass
class DefaultSCRUBUnlearningConfig:
    batch_size: int = SCRUBUnlearning.ORIGINAL_BATCH_SIZE
    gamma: float = SCRUBUnlearning.ORIGNAL_GAMMA
    alpha: float = SCRUBUnlearning.ORIGINAL_ALPHA
    beta: float = SCRUBUnlearning.ORIGINAL_BETA
    smoothing: float = SCRUBUnlearning.ORIGINAL_SMOOTHING
    msteps: int = SCRUBUnlearning.ORIGINAL_MSTEPS
    clip: float = SCRUBUnlearning.ORIGINAL_CLIP
    sstart: int = SCRUBUnlearning.ORIGINAL_SSTART
    kd_t: float = SCRUBUnlearning.ORGIGINAL_KD_T
    learning_rate: float = SCRUBUnlearning.ORIGINAL_LR
    weight_decay: float = SCRUBUnlearning.ORIGINAL_WEIGHT_DECAY
    momentum: float = SCRUBUnlearning.ORIGINAL_MOMENTUM
    forget_batch_size: int = SCRUBUnlearning.ORIGINAL_FORGET_BATCH_SIZE
    lr_decay_rate: float = SCRUBUnlearning.ORIGINAL_LR_DECAY_RATE
    sgda_num_epochs: int = SCRUBUnlearning.ORIGINAL_SGDA_NUM_EPOCHS

    scheduler: typ.Union[typ.Dict[str, typ.Any], None] = None
    criterion: typ.Union[typ.Dict[str, typ.Any], None] = None
    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)
