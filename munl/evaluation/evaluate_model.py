from pandas import DataFrame
from torch.nn import Module
from torch.utils.data import DataLoader

from pipeline.lira.step_1_lira_generate_splits import get_retain_forget_val_test_indices
from munl.datasets import get_loaders_from_dataset_and_unlearner_from_cfg_with_indices
import numpy as np

from munl.settings import default_evaluation_loaders
from munl.evaluation.accuracy import compute_accuracy
from munl.evaluation.common import extract_predictions
from munl.evaluation.membership_inference_attack import evaluate_mia_on_model


def evaluate_model(
    model: Module,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str = "cuda",
) -> DataFrame:
    """
    Evaluate the performance of a model on different datasets.

    Args:
        model (Module): The model to evaluate.
        retain_loader (DataLoader): The data loader for the retain dataset.
        forget_loader (DataLoader): The data loader for the forget dataset.
        val_loader (DataLoader): The data loader for the validation dataset.
        test_loader (DataLoader): The data loader for the test dataset.
        device (str, optional): The device to use for evaluation. Defaults to "cuda".

    Returns:
        DataFrame: A DataFrame containing the evaluation results.
    """
    columns = ("Retain", "Forget", "Val", "Test", "Val MIA", "Test MIA")
    loaders = (retain_loader, forget_loader, val_loader, test_loader)
    row = []
    for loader in loaders:
        y_true, y_pred = extract_predictions(model, loader, device=device)
        accuracy = compute_accuracy(y_true, y_pred)
        row.append(accuracy)
    val_mia = evaluate_mia_on_model(model, val_loader, forget_loader).mean()
    test_mia = evaluate_mia_on_model(model, test_loader, forget_loader).mean()
    row.extend([val_mia, test_mia])
    df = DataFrame([row], columns=columns)
    return df


import argparse
from pathlib import Path

import torch
import torchvision

import munl.models
from munl.configurations import get_img_size_for_dataset, get_num_classes
from munl.datasets import get_loaders_from_dataset_and_unlearner_from_cfg

from munl.models import get_model
from munl.settings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_RANDOM_STATE,
    default_evaluation_loaders,
)
from munl.utils import DictConfig, setup_seed


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--split_ndx", type=int, default=None)
    parser.add_argument("--forget_ndx", type=int, default=None)

    return parser


def evaluate_model_on_dataset(model, dataset, batch_size, random_state, device):
    setup_seed(random_state)
    model.eval()
    model.to(device)
    dataset_cfg = DictConfig({"name": dataset})
    unlearner_cfg = DictConfig(
        {"loaders": default_evaluation_loaders(), "batch_size": batch_size}
    )
    train_loader, retain_loader, forget_loader, val_loader, test_loader = (
        get_loaders_from_dataset_and_unlearner_from_cfg(
            root=Path("."),
            dataset_cfg=dataset_cfg,
            unlearner_cfg=unlearner_cfg,
            random_state=random_state,
        )
    )
    evaluation = evaluate_model(
        model, retain_loader, forget_loader, val_loader, test_loader, device=device
    )
    print(evaluation)
    return evaluation


def evaluate_model_on_loaders(
    model, retain_loader, forget_loader, val_loader, test_loader, random_state, device
):
    setup_seed(random_state)
    model.eval()
    model.to(device)
    print(
        f"Evaluting on loaders of sizes: Retain {len(retain_loader.dataset)} | Forget {len(forget_loader.dataset)} | Val {len(val_loader.dataset)} | Test {len(test_loader.dataset)}"
    )
    print(retain_loader.dataset.indices, forget_loader.dataset.indices, val_loader.dataset.indices, test_loader.dataset.indices)
    evaluation = evaluate_model(
        model, retain_loader, forget_loader, val_loader, test_loader, device=device
    )
    print(evaluation)
    return evaluation


def evaluate_model_from_path_on_dataset(
    model_path, model_type, dataset, batch_size, random_state, device
):
    num_classes = get_num_classes(dataset)
    img_size = get_img_size_for_dataset(dataset)
    model = get_model(model_type, num_classes, img_size)
    weights = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(weights)
    return evaluate_model_on_dataset(
        model_path, model_type, dataset, batch_size, random_state, device
    )


class ModelEvaluationApp:
    def __init__(self, model, dataset, batch_size, random_state, device):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device

    def run(self):
        return evaluate_model_on_dataset(
            model=self.model,
            dataset=self.dataset,
            batch_size=self.batch_size,
            random_state=self.random_state,
            device=self.device,
        )

    def run_on_loaders(self, retain_loader, forget_loader, val_loader, test_loader):
        return evaluate_model_on_loaders(
            model=self.model,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            random_state=self.random_state,
            device=self.device,
        )


class ModelEvaluationFromPathApp(ModelEvaluationApp):
    def __init__(
        self, model_path, model_type, dataset, batch_size, random_state, device
    ):
        num_classes = get_num_classes(dataset)
        img_size = get_img_size_for_dataset(dataset)
        model = get_model(model_type, num_classes, img_size)
        weights = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(weights)
        super().__init__(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            random_state=random_state,
            device=device,
        )


def main(args):
    model_path = Path(args.model_path)
    model_type = args.model_type
    dataset = args.dataset
    batch_size = args.batch_size
    random_state = args.random_state
    device = args.device
    app = ModelEvaluationFromPathApp(
        model_path=model_path,
        model_type=model_type,
        dataset=dataset,
        batch_size=batch_size,
        random_state=random_state,
        device=device,
    )
    if args.split_ndx is None and args.forget_ndx is None:
        app.run()
    else:
        assert args.split_ndx is not None and args.forget_ndx is not None
        root = Path(".")
        retain, forget, val, test = get_retain_forget_val_test_indices(
            lira_path=root / "artifacts" / "lira" / "splits",
            split_ndx=args.split_ndx,
            forget_ndx=args.forget_ndx,
        )
        dataset_cfg = DictConfig({"name": "cifar10"})
        unlearner_cfg = DictConfig(
            {"loaders": default_evaluation_loaders(), "batch_size": 128}
        )

        train_loader, retain_loader, forget_loader, val_loader, test_loader = (
            get_loaders_from_dataset_and_unlearner_from_cfg_with_indices(
                root=root,
                indices=[np.concatenate([retain, forget]), retain, forget, val, test],
                dataset_cfg=dataset_cfg,
                unlearner_cfg=unlearner_cfg,
            )
        )
        app.run_on_loaders(retain_loader, forget_loader, val_loader, test_loader)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
