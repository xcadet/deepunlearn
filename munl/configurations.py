import typing as typ
from dataclasses import dataclass, field
from functools import partial

from hydra_zen import store

import munl.datasets
from munl.settings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MODEL_INIT_DIR,
    DEFAULT_MOMENTUM,
    DEFAULT_TRAINING_EPOCHS,
    DEFAULT_UNLEARN_EPOCHS,
    DEFAULT_WEIGHT_DECAY,
    MODEL_INIT_DIR,
    default_criterion,
    default_loaders,
    default_scheduler,
)
from munl.unlearning import BaseUnlearner


@dataclass
class DatasetConfig:
    name: str = ""
    num_classes: int = 0


@dataclass
class ModelConfig:
    name: str
    variant: str = ""


def get_img_size_for_dataset(dataset_name: str) -> int:
    dataset_to_variant = {
        "mnist": munl.datasets.MNIST_IMAGE_SIZE,
        "fashion_mnist": munl.datasets.FASHION_MNIST_IMAGE_SIZE,
        "cifar10": munl.datasets.CIFAR10_IMAGE_SIZE,
        "cifar100": munl.datasets.CIFAR100_IMAGE_SIZE,
        "utkface": munl.datasets.UTKFACE_IMAGE_SIZE,
    }
    return dataset_to_variant[dataset_name]


@dataclass
class UnlearnerConfig:
    unlearner: BaseUnlearner
    name: str


# Datasets Configurations
# =======================
# Create
mnist_conf = DatasetConfig(name="mnist", num_classes=10)
fashion_mnist_conf = DatasetConfig(name="fashion_mnist", num_classes=10)
cifar10_conf = DatasetConfig(name="cifar10", num_classes=10)
cifar100_conf = DatasetConfig(name="cifar100", num_classes=100)
svhn_conf = DatasetConfig(name="svhn", num_classes=10)
tiny_imagenet_conf = DatasetConfig(name="tiny_imagenet", num_classes=200)

celeba_conf = DatasetConfig(name="celeba", num_classes=2)
utk_face_conf = DatasetConfig(name="utkface", num_classes=5)

# Store
dataset_store = store(group="dataset")
dataset_store(cifar10_conf, name="cifar10")
dataset_store(cifar100_conf, name="cifar100")
dataset_store(mnist_conf, name="mnist")
dataset_store(fashion_mnist_conf, name="fashion_mnist")
dataset_store(celeba_conf, name="celeba")
dataset_store(utk_face_conf, name="utkface")
dataset_store(svhn_conf, name="svhn")
dataset_store(tiny_imagenet_conf, name="tiny_imagenet")

# Models Configurations
# =====================

# Create
resnet18_conf = ModelConfig(name="resnet18")
vit11m_conf = ModelConfig(name="vit11m")

# Store
model_store = store(group="model")
model_store(resnet18_conf, name="resnet18")
model_store(vit11m_conf, name="vit11m")


def default_optimizer():
    return {
        "type": "torch.optim.SGD",
        "learning_rate": DEFAULT_LEARNING_RATE,
        "momentum": DEFAULT_MOMENTUM,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
    }


def loaders_config(state="train", shuffle=False):
    return {"state": state, "shuffle": shuffle}


@dataclass
class DefaultUnlearnerConfig:
    num_epochs: int = DEFAULT_UNLEARN_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    optimizer: typ.Dict[str, typ.Any] = field(default_factory=default_optimizer)
    scheduler: typ.Union[typ.Dict[str, typ.Any], None] = field(
        default_factory=default_scheduler
    )
    criterion: typ.Dict[str, typ.Any] = field(default_factory=default_criterion)
    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)


# Unlearner Configurations
unlearner_store = store(group="unlearner")


def unlearner_config_factory(**kwargs):
    return DefaultUnlearnerConfig(**kwargs)


from munl.settings import augmented_train_retain_forget_loaders

trainer_basic_config = partial(
    unlearner_config_factory,
    num_epochs=DEFAULT_TRAINING_EPOCHS,
    model_initializations_dir=MODEL_INIT_DIR,
    loaders=augmented_train_retain_forget_loaders(),
)
unlearner_basic_config = partial(
    unlearner_config_factory, num_epochs=DEFAULT_UNLEARN_EPOCHS
)


# Training "Unlearners" (Original Model and Retrained Model)
@dataclass
@unlearner_store(name="original")
class OriginalTrainerConf:
    _target_ = "munl.unlearning.OriginalTrainer"
    cfg: DefaultUnlearnerConfig = field(default_factory=trainer_basic_config)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="naive")
class NaiveUnlearnerConf:
    _target_ = "munl.unlearning.NaiveUnlearner"
    cfg: DefaultUnlearnerConfig = field(default_factory=trainer_basic_config)
    device: str = DEFAULT_DEVICE


# Actual Unlearners:


@dataclass
@unlearner_store(name="finetune")
class FinetuneUnlearner:
    _target_: str = (
        "munl.unlearning.FinetuneUnlearner"
    )
    cfg: DefaultUnlearnerConfig = field(default_factory=unlearner_basic_config)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="bad_teacher")
class BadTeacher:
    from munl.unlearning.bad_teacher import DefaultBadTeacherConfig

    _target_: str = "munl.unlearning.BadTeacher"
    cfg: DefaultBadTeacherConfig = field(default_factory=DefaultBadTeacherConfig)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="successive_random_labels")
class SuccesiveRandomLabelsUnlearner:
    _target_: str = (
        "munl.unlearning.SuccessiveRandomLabels"
    )
    cfg: DefaultUnlearnerConfig = field(default_factory=unlearner_basic_config)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="negative_gradient")
class NegativeGradientUnlearner:
    _target_: str = (
        "munl.unlearning.NegativeGradient"
    )
    cfg: DefaultUnlearnerConfig = field(default_factory=unlearner_basic_config)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="gradient_ascent")
class GradientAscentUnlearner:
    from munl.unlearning.gradient_ascent import GradientAscentConfig

    _target_: str = (
        "munl.unlearning.GradientAscent" 
    )
    cfg: GradientAscentConfig = field(default_factory=GradientAscentConfig)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="fisher")
class FisherForgetttinUnlearner:
    from munl.unlearning.fisher import DefaultFisherForgettingConfig

    _target_: str = (
        "munl.unlearning.FisherForgetting"
    )
    cfg: DefaultFisherForgettingConfig = field(
        default_factory=DefaultFisherForgettingConfig
    )
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="salun")
class SalUNUnlearner:
    from munl.unlearning.salun import DefaultSaliencyUnlearningConfig

    _target_: str = (
        "munl.unlearning.SaliencyUnlearning"
    )
    cfg: DefaultSaliencyUnlearningConfig = field(
        default_factory=DefaultSaliencyUnlearningConfig
    )
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="scrub")
class ScrubUnlearner:
    from munl.unlearning.scrub import DefaultSCRUBUnlearningConfig

    _target_: str = (
        "munl.unlearning.SCRUBUnlearning"
    )
    cfg: DefaultSCRUBUnlearningConfig = field(
        default_factory=DefaultSCRUBUnlearningConfig
    )
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="scrubv2")
class ScrubUnlearnerV2:
    from munl.unlearning.scrubv2 import DefaultSCRUBUnlearningV2Config

    _target_: str = (
        "munl.unlearning.SCRUBUnlearningV2"
    )
    cfg: DefaultSCRUBUnlearningV2Config = field(
        default_factory=DefaultSCRUBUnlearningV2Config
    )
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="neggradplus")
class NegGradPlusUnlearner:
    from munl.unlearning.neggradplus import DefaultNegGradPlusUnlearningConfig

    _target_: str = "munl.unlearning.NegGradPlus"
    cfg: DefaultNegGradPlusUnlearningConfig = field(
        default_factory=DefaultNegGradPlusUnlearningConfig
    )
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="influence")
class InfluenceUnlearner:
    from munl.unlearning.iu import DefaultInfluenceUnlearningConfig

    _target_: str = (
        "munl.unlearning.InfluenceUnlearning"
    )
    cfg: DefaultInfluenceUnlearningConfig = field(
        default_factory=DefaultInfluenceUnlearningConfig
    )
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="cfk")
class CFKUnlearner:
    from munl.unlearning.catastrophic_forgetting_k import (
        DefaultCatastrophicForgettingKConfig,
    )

    _target_: str = (
        "munl.unlearning.CatastrophicForgettingK"
    )
    cfg: DefaultCatastrophicForgettingKConfig = field(
        default_factory=DefaultCatastrophicForgettingKConfig
    )
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="euk")
class EUKNUnlearner:
    from munl.unlearning.exact_unlearning_k import DefaultExactUnlearningKConfig

    _target_: str = (
        "munl.unlearning.ExactUnlearningK"
    )
    cfg: DefaultExactUnlearningKConfig = field(
        default_factory=DefaultExactUnlearningKConfig
    )
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="kgltop1")
class KGlTop1Unlearner:
    from munl.unlearning.kgltop1 import DefaultKGLTop1Config

    _target_: str = "munl.unlearning.KGLTop1"
    cfg: DefaultKGLTop1Config = field(default_factory=DefaultKGLTop1Config)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="kgltop2")
class KGLTop2Unlearner:
    from munl.unlearning.kgltop2 import DefaultKGLTop2Config

    _target_: str = "munl.unlearning.KGLTop2"
    cfg: DefaultKGLTop2Config = field(default_factory=DefaultKGLTop2Config)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="kgltop3")
class KGLTop3Unlearner:
    from munl.unlearning.kgltop3 import DefaultKGLTop3Config

    _target_: str = "munl.unlearning.KGLTop3"
    cfg: DefaultKGLTop3Config = field(default_factory=DefaultKGLTop3Config)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="kgltop4")
class KGLTop4Unlearner:
    from munl.unlearning.kgltop4 import DefaultKGLTop4Config

    _target_: str = "munl.unlearning.KGLTop4"
    cfg: DefaultKGLTop4Config = field(default_factory=DefaultKGLTop4Config)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="kgltop5")
class KGLTop5Unlearner:
    from munl.unlearning.kgltop5 import DefaultKGLTop5Config

    _target_: str = "munl.unlearning.KGLTop5"
    cfg: DefaultKGLTop5Config = field(default_factory=DefaultKGLTop5Config)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="kgltop6")
class KGLTop6Unlearner:
    from munl.unlearning.kgltop6 import DefaultKGLTop6Config

    _target_: str = "munl.unlearning.KGLTop6"
    cfg: DefaultKGLTop6Config = field(default_factory=DefaultKGLTop6Config)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="kgltop7")
class KGLTop7Unlearner:
    from munl.unlearning.kgltop7 import DefaultKGLTop7Config

    _target_: str = "munl.unlearning.KGLTop7"
    cfg: DefaultKGLTop7Config = field(default_factory=DefaultKGLTop7Config)
    device: str = DEFAULT_DEVICE


def get_dataset_config(dataset_name) -> DatasetConfig:
    datasets = dataset_store["dataset"]
    matching = list(filter(lambda dataset: dataset[1] == dataset_name, datasets))
    assert len(matching) <= 1
    return datasets[matching[0]] if len(matching) == 1 else None


def get_num_classes(dataset_name: str) -> int:
    # Retrieve the configuration from the dataset store
    dataset_config = get_dataset_config(dataset_name)
    assert dataset_config is not None, f"Dataset {dataset_name} not found"
    return dataset_config.num_classes
