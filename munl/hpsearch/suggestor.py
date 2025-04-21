import math
from typing import Dict, Union, Optional

from omegaconf import DictConfig
from optuna.trial import Trial

from munl.settings import (
    HP_NUM_EPOCHS,
    HP_NUM_EPOCHS_FLOAT,
    HP_BATCH_SIZE,
    HP_OPTUNA,
    TrialSuggestionFunctionType,
)


def set_nested_value(obj: DictConfig, dot_key: str, value: Union[int, float]) -> None:
    """Sets a nested value in an object or dictionary."""
    keys = dot_key.split(".")
    for key in keys[:-1]:
        if hasattr(obj, key):
            obj = getattr(obj, key)
        elif isinstance(obj, dict) and key in obj:
            obj = obj[key]
        else:
            raise AttributeError(f"Key '{key}' not found in object or dictionary.")

    if hasattr(obj, keys[-1]):
        setattr(obj, keys[-1], value)
    elif isinstance(obj, dict):
        if keys[-1] in obj:
            obj[keys[-1]] = value
        else:
            raise KeyError(f"Key '{keys[-1]}' not found in dictionary.")
    else:
        raise AttributeError(f"Key '{keys[-1]}' not found in object.")


class HyperParameterSuggestor:
    def __init__(
        self,
        dataset: str,
        lira: bool,
        hp_type_to_suggestor: Dict[str, TrialSuggestionFunctionType] = HP_OPTUNA,
    ):
        """ """
        self.dataset = dataset
        self.hp_type_to_suggestor = hp_type_to_suggestor
        self.lira = lira

    def get_num_epochs_suggestion(self) -> int:
        percentage: float = 0.2
        if self.dataset in ["cifar10", "cifar100"]:
            epochs = 91 if self.lira else 182
            return math.ceil(epochs * percentage)
        elif self.dataset in ["mnist", "fashion_mnist", "utkface"]:
            return math.ceil(50 * percentage)
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not supported.")

    def get_batch_size_suggestion(self) -> Optional[int]:
        if self.dataset in ["utkface"]:
            return 128  # Maximal batch_size
        return None

    def suggest_in_place(
        self, cfg: DictConfig, hyper_parameters: Dict[str, str], trial: Trial
    ):
        """Suggests hyper-parameters in-place for the given configuration.

        Args:
            cfg (DictConfig): The configuration dictionary.
            hyper_parameters (Dict[str, callable]): A dictionary of hyper-parameter
                                                    paths and their corresponding types.
            trial (Trial): The Optuna trial object.

        Returns:
            DictConfig: The updated configuration dictionary with suggested
                        hyper-parameters.
        """
        value: Union[int, float]
        func: TrialSuggestionFunctionType
        for hp_path, hp_type in hyper_parameters.items():
            assert hp_path.startswith(
                "unlearner.cfg"
            ), "Issue with the hyper parameters"
            assert isinstance(hp_type, str), "Issue with the hyper parameters type"
            new_key = hp_path.replace("unlearner.cfg.", "")
            func = self.hp_type_to_suggestor[hp_type]
            if is_epoch_suggestion(hp_type):
                value = func(trial, new_key, self.get_num_epochs_suggestion())
            else:
                value = func(trial, new_key)
            set_nested_value(cfg, new_key, value)
        return cfg


def is_epoch_suggestion(hp_type: str) -> bool:
    """Determines if the hyper-parameter type is an epoch suggestion."""
    return hp_type in [HP_NUM_EPOCHS, HP_NUM_EPOCHS_FLOAT]


def is_batch_size_suggestion(hp_type: str) -> bool:
    """Determines if the hyper-parameter type is an epoch suggestion."""
    return hp_type in [HP_BATCH_SIZE]
