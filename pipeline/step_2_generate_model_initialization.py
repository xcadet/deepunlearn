import argparse
import logging
from pathlib import Path

from hydra_zen import make_config, store

from munl import MODEL_INITIALIZATIONS_PATH
from munl.configurations import ModelConfig
from munl.models import get_model, save_model_state_dict
from munl.settings import DEFAULT_RANDOM_STATE
from munl.utils import setup_seed

logger = logging.getLogger(__name__)

store(
    make_config(
        hydra_defaults=["_self_", {"model": "resnet18"}],
        model=None,
        num_classes=10,
        output_path=MODEL_INITIALIZATIONS_PATH,
        model_seed=0,
        img_size=32,
    ),
    name="generate_model_initialization",
)


def zen_generate_model_initialization(
    model: ModelConfig,
    num_classes: int,
    output_path: str,
    model_seed: int,
    img_size: int,
):
    return generate_model_initialization(
        model.name, num_classes, Path(output_path), model_seed, img_size
    )


def generate_model_initialization(
    model_name: str,
    num_classes: int,
    output_path: Path,
    model_seed: int,
    img_size: int,
):
    setup_seed(model_seed)
    model = get_model(model_name, num_classes, img_size)
    save_model_state_dict(
        model, output_path, num_classes, model_name, model_seed, img_size=img_size
    )


if __name__ == "__main__":
    from hydra.conf import HydraConf, JobConf
    from hydra_zen import zen

    store(HydraConf(job=JobConf(chdir=False)), name="config", group="hydra")
    store.add_to_hydra_store()
    zen(zen_generate_model_initialization).hydra_main(
        config_name="generate_model_initialization",
        version_base="1.1",
        config_path=None,
    )
