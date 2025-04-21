import copy
import os
import typing as typ
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from torch.utils.data import DataLoader

from thirdparty.repdistiller.distiller_zoo import DistillKL
from thirdparty.repdistiller.helper.loops import train_distill, validate
from thirdparty.repdistiller.helper.util import (
    adjust_learning_rate as sgda_adjust_learning_rate,
)


def generate_unique_identifier():
    job_id = os.getenv("SLURM_JOB_ID", "no_job_id")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    random_uuid = uuid.uuid4().hex

    unique_id = f"{job_id}_{timestamp}_{random_uuid}"

    return unique_id


import time
from argparse import Namespace

from omegaconf import DictConfig

import munl.settings as munl_hp
from munl.datasets import (
    extract_targets_only,
    is_shuffling,
    update_dataloader_batch_size,
)
from munl.settings import DEFAULT_MODEL_INIT_DIR, default_loaders_no_shuffle_forget
from munl.unlearning import BaseUnlearner


def swa_avg_fn(beta: float, averaged_model_parameter, model_parameter):
    return (1 - beta) * averaged_model_parameter + beta * model_parameter


# NOTE: from https://github.com/meghdadk/SCRUB/blob/812dd5deae230eed7c82e25f91d66d63d877ec7c/MIA_experiments.ipynb # noqa
class SCRUBUnlearningV2(BaseUnlearner):
    ORIGINAL_OPTIM = "sgd"
    ORIGINAL_GAMMA = 1  
    ORIGINAL_ALPHA = 0.5  
    ORIGINAL_BETA = 0  
    ORIGINAL_SMOOTHING = 0.5 
    ORIGINAL_DISTILL = "kd"  

    ORIGINAL_MSTEPS = 3  
    ORIGINAL_KD_T = 2  

    ORIGINAL_SGDA_EPOCHS = 5  
    ORIGINAL_SGDA_LEARNING_RATE = 0.0005  
    ORIGINAL_LR_DECAY_EPOCHS = [
        3,
        5,
        9,
    ]  
    ORIGINAL_LR_DECAY_RATE = (
        0.1  
    )
    ORIGINAL_SGDA_WEIGHT_DECAY = 5e-4  
    ORIGINAL_SGDA_MOMENTUM = 0.9  


    ORIGINAL_RETAIN_BATCH_SIZE = 32  
    ORIGINAL_FORGET_BATCH_SIZE = 64  

    HYPER_PARAMETERS = {
        "unlearner.cfg.gamma": munl_hp.HP_FLOAT,
        "unlearner.cfg.alpha": munl_hp.HP_FLOAT,
        "unlearner.cfg.beta": munl_hp.HP_FLOAT,
        "unlearner.cfg.smoothing": munl_hp.HP_FLOAT,
        "unlearner.cfg.msteps": munl_hp.HP_NUM_EPOCHS,
        "unlearner.cfg.kd_T": munl_hp.HP_TEMPERATURE,
        "unlearner.cfg.sgda_num_epochs": munl_hp.HP_NUM_EPOCHS,
        "unlearner.cfg.sgda_learning_rate": munl_hp.HP_LEARNING_RATE,
        "unlearner.cfg.sgda_weight_decay": munl_hp.HP_WEIGHT_DECAY,
        "unlearner.cfg.sgda_momentum": munl_hp.HP_MOMENTUM,
        "unlearner.cfg.retain_batch_size": munl_hp.HP_BATCH_SIZE,
        "unlearner.cfg.forget_batch_size": munl_hp.HP_BATCH_SIZE,
    }

    def __init__(
        self,
        cfg: DictConfig,
        device: str,
        writer=None,
        save_steps: bool = False,
        should_evaluate: bool = False,
    ):
        super().__init__(
            cfg=cfg,
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
        unique_id = generate_unique_identifier()
        device = self.device
        sgda_epochs = self.cfg.sgda_num_epochs  # noqa
        lr_decay_epochs = [
            int(sgda_epochs * 0.5),
            int(sgda_epochs * 0.8),
            int(sgda_epochs * 0.9),
        ]
        sgda_learning_rate = self.cfg.sgda_learning_rate  # noqa
        sgda_momentum = self.cfg.sgda_momentum  # noqa
        lr_decay_rate = self.cfg.lr_decay_rate  # noqa
        sgda_weight_decay = self.cfg.sgda_weight_decay  # noqa
        kd_T = self.cfg.kd_T  # noqa
        msteps = self.cfg.msteps  # noqa
        retain_batch_size = self.cfg.retain_batch_size  # noqa
        forget_batch_size = self.cfg.forget_batch_size  # noqa

        opt = Namespace()
        opt.distill = "kd"
        opt.smoothing = self.cfg.smoothing
        opt.gamma = self.cfg.gamma
        opt.beta = self.cfg.beta
        opt.alpha = self.cfg.alpha

        retain_loader = update_dataloader_batch_size(retain_loader, retain_batch_size)
        forget_loader = update_dataloader_batch_size(forget_loader, forget_batch_size)

        model.to(device)
        model_t = copy.deepcopy(model)
        model_s = copy.deepcopy(model)

        module_list = nn.ModuleList([])
        module_list.append(model_s)
        trainable_list = nn.ModuleList([])
        trainable_list.append(model_s)

        criterion_cls = nn.CrossEntropyLoss()
        criterion_div = DistillKL(kd_T)
        criterion_kd = DistillKL(kd_T)

        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_cls)  # classification loss
        criterion_list.append(
            criterion_div
        )  # KL divergence loss, original knowledge distillation
        criterion_list.append(criterion_kd)  # other knowledge distillation loss

        optimizer = optim.SGD(
            trainable_list.parameters(),
            lr=sgda_learning_rate,
            momentum=sgda_momentum,
            weight_decay=sgda_weight_decay,
        )
        module_list.append(model_t)

        if torch.cuda.is_available():
            module_list.cuda()
            criterion_list.cuda()
            import torch.backends.cudnn as cudnn

            cudnn.benchmark = True

        t1 = time.time()
        acc_rs = []
        acc_fs = []
        acc_vs = []
        acc_fvs = []

        assert not is_shuffling(val_loader), "Validation loader must not shuffle"
        assert not is_shuffling(val_loader), "Forget loader must not shuffle"
        forget_validation_loader = copy.deepcopy(val_loader)

        setattr(
            forget_validation_loader.dataset,
            "targets",
            extract_targets_only(forget_validation_loader),
        )
        setattr(
            forget_loader.dataset,
            "targets",
            extract_targets_only(forget_loader),
        )

        # Find the unique classes in the forget_loader
        fgt_cls = list(np.unique(forget_loader.dataset.targets))

        # Generate indices for the classes to forget
        indices = [i in fgt_cls for i in forget_validation_loader.dataset.targets]

        # If the dataset is a Subset, access the underlying dataset and its indices
        if isinstance(forget_validation_loader.dataset, torch.utils.data.Subset):
            subset_indices = forget_validation_loader.dataset.indices
            # underlying_dataset = forget_validation_loader.dataset.dataset

            # Filter indices and update the Subset's indices
            filtered_indices = [
                subset_indices[i] for i in range(len(indices)) if indices[i]
            ]
            forget_validation_loader.dataset.indices = filtered_indices
        else:
            # Update the dataset directly if it is not a Subset
            forget_validation_loader.dataset.data = (
                forget_validation_loader.dataset.data[indices]
            )
            forget_validation_loader.dataset.targets = (
                forget_validation_loader.dataset.targets[indices]
            )
        scrub_name = Path(f"checkpoints/{unique_id}_scrub_step")
        scrub_name.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, sgda_epochs + 1):

            lr = sgda_adjust_learning_rate(
                epoch=epoch,
                lr_decay_epochs=lr_decay_epochs,
                sgda_learning_rate=sgda_learning_rate,
                lr_decay_rate=lr_decay_rate,
                optimizer=optimizer,
            )
            print(f"Updated optimizer learning rate to '{lr}'")
            validate_opt = Namespace()
            validate_opt.print_freq = 0

            acc_r, acc5_r, loss_r = validate(
                retain_loader, model_s, criterion_cls, validate_opt, True
            )
            acc_f, acc5_f, loss_f = validate(
                forget_loader, model_s, criterion_cls, validate_opt, True
            )
            acc_v, acc5_v, loss_v = validate(
                val_loader, model_s, criterion_cls, validate_opt, True
            )
            acc_fv, acc5_fv, loss_fv = validate(
                forget_validation_loader, model_s, criterion_cls, validate_opt, True
            )
            acc_rs.append(100 - acc_r.item())
            acc_fs.append(100 - acc_f.item())
            acc_vs.append(100 - acc_v.item())
            acc_fvs.append(100 - acc_fv.item())

            maximize_loss = 0
            if epoch <= msteps:
                maximize_loss = train_distill(
                    epoch,
                    forget_loader,
                    module_list,
                    model_s,
                    criterion_list,
                    optimizer,
                    opt,
                    "maximize",
                )
            train_acc, train_loss = train_distill(
                epoch,
                retain_loader,
                module_list,
                model_s,
                criterion_list,
                optimizer,
                opt,
                "minimize",
            )

            torch.save(model_s.state_dict(), str(scrub_name) + str(epoch) + ".pt")

            print(
                "maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(
                    maximize_loss, train_loss, train_acc
                )
            )
        t2 = time.time()
        print(t2 - t1)

        acc_r, acc5_r, loss_r = validate(
            retain_loader, model_s, criterion_cls, validate_opt, True
        )
        acc_f, acc5_f, loss_f = validate(
            forget_loader, model_s, criterion_cls, validate_opt, True
        )
        acc_v, acc5_v, loss_v = validate(
            val_loader, model_s, criterion_cls, validate_opt, True
        )
        acc_fv, acc5_fv, loss_fv = validate(
            forget_validation_loader, model_s, criterion_cls, validate_opt, True
        )
        acc_rs.append(100 - acc_r.item())
        acc_fs.append(100 - acc_f.item())
        acc_vs.append(100 - acc_v.item())
        acc_fvs.append(100 - acc_fv.item())

        try:
            selected_idx, _ = min(
                enumerate(acc_fs), key=lambda x: abs(x[1] - acc_fvs[-1])
            )
        except Exception as e:
            if len(acc_fs) == 1:
                selected_idx = 1
            else:
                selected_idx = len(acc_fs) - 1
            print("Exception: {}".format(e))
        print("the selected index is {}".format(selected_idx))
        if selected_idx != 0:
            selected_model = "checkpoints/{}_scrub_step{}.pt".format(
                unique_id, int(selected_idx)
            )
            model_s.load_state_dict(torch.load(selected_model))
        return model_s  


@dataclass
class DefaultSCRUBUnlearningV2Config:
    # We need to add the batch_size as default
    batch_size: int = SCRUBUnlearningV2.ORIGINAL_RETAIN_BATCH_SIZE
    gamma: float = SCRUBUnlearningV2.ORIGINAL_GAMMA
    alpha: float = SCRUBUnlearningV2.ORIGINAL_ALPHA
    beta: float = SCRUBUnlearningV2.ORIGINAL_BETA
    smoothing: float = SCRUBUnlearningV2.ORIGINAL_SMOOTHING
    msteps: int = SCRUBUnlearningV2.ORIGINAL_MSTEPS
    kd_T: float = SCRUBUnlearningV2.ORIGINAL_KD_T
    sgda_num_epochs: int = SCRUBUnlearningV2.ORIGINAL_SGDA_EPOCHS
    sgda_learning_rate: float = SCRUBUnlearningV2.ORIGINAL_SGDA_LEARNING_RATE
    sgda_weight_decay: float = SCRUBUnlearningV2.ORIGINAL_SGDA_WEIGHT_DECAY
    sgda_momentum: float = SCRUBUnlearningV2.ORIGINAL_SGDA_MOMENTUM
    retain_batch_size: int = SCRUBUnlearningV2.ORIGINAL_RETAIN_BATCH_SIZE
    forget_batch_size: int = SCRUBUnlearningV2.ORIGINAL_FORGET_BATCH_SIZE
    lr_decay_rate: float = SCRUBUnlearningV2.ORIGINAL_LR_DECAY_RATE

    scheduler: typ.Union[typ.Dict[str, typ.Any], None] = None
    criterion: typ.Union[typ.Dict[str, typ.Any], None] = None
    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(
        default_factory=default_loaders_no_shuffle_forget
    )
