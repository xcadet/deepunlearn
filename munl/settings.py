from typing import Callable, Dict, Union

import matplotlib
from matplotlib import colormaps
from optuna.trial import Trial
from collections import OrderedDict


def generate_colors_from_colormap(num_colors, cmap_name="viridis"):
    """
    Generate a list of colors from a specified Matplotlib colormap.

    Args:
    num_colors (int): Number of colors to generate.
    cmap_name (str): Name of the colormap to use.

    Returns:
    list: A list of colors in HEX format.
    """
    colormap = colormaps[cmap_name]  # Get the colormap
    colors = [colormap(i) for i in range(colormap.N)]  # Extract the colors as RGBA
    step = len(colors) // num_colors  # Determine step to get evenly spaced colors

    # Select colors at evenly spaced intervals
    selected_colors = [colors[i * step] for i in range(num_colors)]

    # Convert RGBA to HEX (skip if you prefer RGBA)
    hex_colors = [matplotlib.colors.to_hex(color[:3]) for color in selected_colors]

    return hex_colors


MARKERS = [
    "o",
    "s",
    "D",
    "^",
    "v",
    "<",
    ">",
    "p",
    "*",
    "h",
    "H",
    "+",
    "x",
    "d",
    "|",
    "_",
    "8",
    "1",
    "2",
    "3",
]


MODEL_INIT_DIR = "model_initializations/"
DEFAULT_MODEL_INIT_DIR = "unlearn/unlearner_original"
# DEFAULT_DEVICE = "cpu"
DEFAULT_DEVICE = "cuda"

DEFAULT_OPTUNA_N_TRIALS = 100
DEFAULT_TRAINING_EPOCHS = 200
DEFAULT_UNLEARN_EPOCHS = 5
# Batch size should be 64
# but updated with values from https://arxiv.org/pdf/2310.12508.pdf
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 0.1  # https://arxiv.org/pdf/2304.04934.pdf
DEFAULT_MOMENTUM = 0.9  # https://arxiv.org/pdf/2304.04934.pdf
DEFAULT_WEIGHT_DECAY = 5e-4  # https://arxiv.org/pdf/2304.04934.pdf
DEFAULT_RANDOM_STATE = 123
DEFAULT_PIN_MEMORY = False


TRAIN_STATE = "train"
TEST_STATE = "test"


def augmented_train_retain_forget_loaders():
    res = {
        "train": {"state": TRAIN_STATE, "shuffle": True},
        "retain": {"state": TRAIN_STATE, "shuffle": True},
        "forget": {"state": TRAIN_STATE, "shuffle": True},
        "val": {"state": TEST_STATE, "shuffle": False},
        "test": {"state": TEST_STATE, "shuffle": False},
    }
    return res


def augmented_train_retain_loaders():
    res = {
        "train": {"state": TRAIN_STATE, "shuffle": True},
        "retain": {"state": TRAIN_STATE, "shuffle": True},
        "forget": {"state": TEST_STATE, "shuffle": True},
        "val": {"state": TEST_STATE, "shuffle": False},
        "test": {"state": TEST_STATE, "shuffle": False},
    }
    return res


def default_loaders():
    return {
        "train": {"state": TEST_STATE, "shuffle": True},
        "retain": {"state": TEST_STATE, "shuffle": True},
        "forget": {"state": TEST_STATE, "shuffle": True},
        "val": {"state": TEST_STATE, "shuffle": False},
        "test": {"state": TEST_STATE, "shuffle": False},
    }


def default_loaders_no_shuffle_forget():
    default = default_loaders()
    default["forget"]["shuffle"] = False
    return default


def default_evaluation_loaders():
    return {
        "train": {"state": TEST_STATE, "shuffle": False},
        "retain": {"state": TEST_STATE, "shuffle": False},
        "forget": {"state": TEST_STATE, "shuffle": False},
        "val": {"state": TEST_STATE, "shuffle": False},
        "test": {"state": TEST_STATE, "shuffle": False},
    }


def default_scheduler():
    return {
        "type": "torch.optim.lr_scheduler.CosineAnnealingLR",
        "optimizer": None,
        "T_max": None,
    }


def default_criterion():
    return {
        "type": "torch.nn.CrossEntropyLoss",
    }


# Hyper parameters mapping and their associated sweeps
HP_LEARNING_RATE = "learning_rate"
HP_WEIGHT_DECAY = "weight_decay"
HP_MOMENTUM = "momentum"
HP_NUM_EPOCHS = "num_epochs"
HP_NUM_EPOCHS_FLOAT = "num_epochs_float"
HP_BATCH_SIZE = "batch_size"
HP_NORMAL_SIGMA = "normal_sigma"
HP_FLOAT = "float"
HP_TEMPERATURE = "temperature"
HP_ETA_MIN = "eta_min"
HP_NUM_LAYERS = "num_layers"
HP_TRAINING_EPOCH_FACTOR = "training_epoch_factor"


def optuna_suggest_learning_rate(trial: Trial, name: str) -> float:
    return trial.suggest_float(name, 1e-5, 1e-1, log=True)


def optuna_suggest_num_layers(trial: Trial, name: str, high: int = 10) -> int:
    return trial.suggest_int(name, 1, high)


def optuna_suggest_weight_decay(trial: Trial, name: str) -> float:
    return trial.suggest_float(name, 1e-5, 1e-1, log=True)


def optuna_suggest_momentum(trial: Trial, name: str) -> float:
    return trial.suggest_float(name, low=DEFAULT_MOMENTUM, high=DEFAULT_MOMENTUM)


def optuna_suggest_num_epochs(trial: Trial, name: str, high: int) -> int:
    return trial.suggest_int(name, 1, high)


def optuna_suggest_num_epochs_float(trial: Trial, name: str, high: int) -> float:
    return trial.suggest_float(name, 1, high)


def optuna_suggest_batch_size(trial: Trial, name: str) -> int:
    return trial.suggest_categorical(name, (64, 128, 256))


def optuna_suggest_normal_sigma(trial: Trial, name: str) -> float:
    return trial.suggest_float(name, 0.0, 1.0)


def optuna_suggest_float(trial: Trial, name: str) -> float:
    return trial.suggest_float(name, 0.0, 1.0)


def optuna_suggest_temperature(trial: Trial, name: str) -> float:
    return trial.suggest_float(name, 0.0, 5.0)


def optuna_suggest_eta_min(trial: Trial, name: str) -> float:
    return trial.suggest_float(name, 1e-6, 1e-1, log=True)


def optuna_suggest_trainig_epoch_factor(trial: Trial, name: str) -> int:
    return trial.suggest_int(name, 1, 10)


TrialSuggestionFunctionType = Callable[..., Union[int, float]]


HP_OPTUNA: Dict[str, TrialSuggestionFunctionType] = {
    HP_LEARNING_RATE: optuna_suggest_learning_rate,
    HP_WEIGHT_DECAY: optuna_suggest_weight_decay,
    HP_MOMENTUM: optuna_suggest_momentum,
    HP_NUM_EPOCHS: optuna_suggest_num_epochs,
    HP_NUM_EPOCHS_FLOAT: optuna_suggest_num_epochs_float,
    HP_BATCH_SIZE: optuna_suggest_batch_size,
    HP_NORMAL_SIGMA: optuna_suggest_normal_sigma,
    HP_FLOAT: optuna_suggest_float,
    HP_TEMPERATURE: optuna_suggest_temperature,
    HP_ETA_MIN: optuna_suggest_eta_min,
    HP_NUM_LAYERS: optuna_suggest_num_layers,
    HP_TRAINING_EPOCH_FACTOR: optuna_suggest_trainig_epoch_factor,
}

HYPER_PARAMETERS = {
    "unlearner.cfg.optimizer.learning_rate": HP_LEARNING_RATE,
    "unlearner.cfg.optimizer.momentum": HP_MOMENTUM,
    "unlearner.cfg.optimizer.weight_decay": HP_WEIGHT_DECAY,
    "unlearner.cfg.batch_size": HP_BATCH_SIZE,
    "unlearner.cfg.num_epochs": HP_NUM_EPOCHS,
}


METHODS_TO_READABLE = OrderedDict(
    {
        "naive": "R",
        "original": "O",
        "finetune": "FT",
        "gradient_ascent": "GA",
        "successive_random_labels": "SRL",
        "kgltop1": "FCS",
        "kgltop2": "MSG",
        "kgltop3": "CFW",
        "kgltop4": "PRMQ",
        "kgltop5": "CT",
        "kgltop6": "KDE",
        "kgltop7": "RNI",
        "bad_teacher": "BT",
        "salun": "SalUN",
        "scrub": "SCRUBOLD",
        "scrubv2": "SCRUB",
        "influence": "IU",
        "fisher": "FF",
        "cfk": "CF-k",
        "euk": "EU-k",
        "neggradplus": "NG+",
    }
)

COLORS = generate_colors_from_colormap(20, "tab20")

METHODS_TO_COLOR = OrderedDict(
    {
        "naive": COLORS[0],
        "original": COLORS[1],
        "finetune": COLORS[2],
        "gradient_ascent": COLORS[3],
        "successive_random_labels": COLORS[4],
        "kgltop1": COLORS[5],
        "kgltop2": COLORS[6],
        "kgltop3": COLORS[7],
        "kgltop4": COLORS[8],
        "kgltop5": COLORS[9],
        "kgltop6": COLORS[10],
        "kgltop7": COLORS[11],
        "bad_teacher": COLORS[12],
        "salun": COLORS[13],
        "scrub": COLORS[14],
        "scrubv2": COLORS[14],
        "influence": COLORS[15],
        "fisher": COLORS[16],
        "cfk": COLORS[17],
        "euk": COLORS[18],
        "neggradplus": COLORS[19],
    }
)

DATASET_TO_READABLE = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "mnist": "MNIST",
    "fashion_mnist": "FashionMNIST",
    "utkface": "UTKFace",
}

METHODS_TO_MARKER = OrderedDict(
    {
        "naive": MARKERS[0],
        "original": MARKERS[1],
        "finetune": MARKERS[2],
        "gradient_ascent": MARKERS[3],
        "successive_random_labels": MARKERS[4],
        "kgltop1": MARKERS[5],
        "kgltop2": MARKERS[6],
        "kgltop3": MARKERS[7],
        "kgltop4": MARKERS[8],
        "kgltop5": MARKERS[9],
        "kgltop6": MARKERS[10],
        "kgltop7": MARKERS[11],
        "bad_teacher": MARKERS[12],
        "salun": MARKERS[13],
        "scrub": MARKERS[14],
        "scrubv2": MARKERS[14],
        "influence": MARKERS[15],
        "fisher": MARKERS[16],
        "cfk": MARKERS[17],
        "euk": MARKERS[18],
        "neggradplus": MARKERS[19],
    }
)

ARCHITECTURE_TO_READABLE = {
    "resnet18": "ResNet-18",
    "vit11m": "TinyViT",
}


METHODS = list(METHODS_TO_READABLE.keys())

RESULTS_ROUND = 3
