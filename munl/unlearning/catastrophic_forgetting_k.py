# Adapted from https://github.com/meghdadk/SCRUB/blob/main/small_scale_unlearning.ipynb
import typing as typ
from dataclasses import dataclass, field

from torch.nn import Module
from torch.utils.data import DataLoader

import munl.settings
from munl.models import (
    freeze_model,
    get_model_classifier,
    select_last_k_blocks,
    unfreeze_modules,
)
from munl.settings import DEFAULT_MODEL_INIT_DIR, default_loaders
from munl.unlearning.common import BaseUnlearner
from munl.unlearning.scrub_utils import fk_finetune
from munl.utils import DictConfig, get_num_classes_from_model


class CatastrophicForgettingK(BaseUnlearner):
    # Specific to the original implementation

    # Generic Hyper
    ORIGINAL_SGDA_LR = 0.0005
    ORIGINAL_MOMENTUM = 0.9
    ORIGINAL_WEIGHT_DECAY = 0
    ORIGINAL_NUM_EPOCHS = 10
    ORIGINAL_BATCH_SIZE = 256
    ORIGINAL_LR_DECAY_RATE = 0.1
    ORIGINAL_NUM_BLOCKS = 1
    ORIGINAL_SGDA_WEIGHT_DECAY = 0.1

    # Generic Hyper
    ORIGINAL_LR = 0.01
    ORIGINAL_MOMENTUM = 0.9
    ORIGINAL_WEIGHT_DECAY = 0.1

    HYPER_PARAMETERS = {
        # **settings.HYPER_PARAMETERS,
        "unlearner.cfg.num_epochs": munl.settings.HP_NUM_EPOCHS,
        "unlearner.cfg.batch_size": munl.settings.HP_BATCH_SIZE,
        "unlearner.cfg.sgda_decay_rate": munl.settings.HP_FLOAT,
        "unlearner.cfg.sgda_learning_rate": munl.settings.HP_LEARNING_RATE,
        "unlearner.cfg.sgda_weight_decay": munl.settings.HP_WEIGHT_DECAY,
        "unlearner.cfg.learning_rate": munl.settings.HP_LEARNING_RATE,
        "unlearner.cfg.momentum": munl.settings.HP_MOMENTUM,
        "unlearner.cfg.num_blocks": munl.settings.HP_NUM_LAYERS,
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
        num_epochs = self.cfg.num_epochs
        lr = self.cfg.learning_rate
        sgda_epochs = self.cfg.num_epochs
        num_blocks = self.cfg.num_blocks
        lr_decay_epochs = [
            int(sgda_epochs * 0.5),
            int(sgda_epochs * 0.8),
            int(sgda_epochs * 0.9),
        ]  # Based on the original values
        sgda_learning_rate = self.cfg.learning_rate
        lr_decay_rate = self.cfg.sgda_decay_rate
        sgda_weight_decay = self.cfg.sgda_weight_decay
        num_classes = get_num_classes_from_model(model)

        model.to(device)
        # 1. Model deactivation
        freeze_model(model)
        # 2. Reactivate the k last layers
        blocks_to_unfreeze = select_last_k_blocks(model, num_blocks)
        # 3. We need to add the classification layer
        blocks_to_unfreeze += [get_model_classifier(model)]
        unfreeze_modules(blocks_to_unfreeze)
        # Model finetuning
        fk_finetune(
            model=model,
            data_loader=retain_loader,
            epochs=num_epochs,
            lr=lr,
            num_classes=num_classes,
            weight_decay=sgda_weight_decay,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_rate=lr_decay_rate,
            sgda_learning_rate=sgda_learning_rate,
        )

        return model


@dataclass
class DefaultCatastrophicForgettingKConfig:
    num_epochs: int = CatastrophicForgettingK.ORIGINAL_NUM_EPOCHS
    batch_size: int = CatastrophicForgettingK.ORIGINAL_BATCH_SIZE
    num_blocks: int = CatastrophicForgettingK.ORIGINAL_NUM_BLOCKS
    learning_rate: float = CatastrophicForgettingK.ORIGINAL_LR
    sgda_learning_rate: float = CatastrophicForgettingK.ORIGINAL_SGDA_LR
    sgda_decay_rate: float = CatastrophicForgettingK.ORIGINAL_LR_DECAY_RATE
    sgda_weight_decay: float = CatastrophicForgettingK.ORIGINAL_SGDA_WEIGHT_DECAY
    momentum: float = CatastrophicForgettingK.ORIGINAL_MOMENTUM

    scheduler: typ.Union[typ.Dict[str, typ.Any], None] = None
    criterion: typ.Union[typ.Dict[str, typ.Any], None] = None
    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)
