import argparse
from pipeline.lira.step_1_lira_generate_splits import get_retain_forget_val_test_indices
from pipeline.step_5_unlearn import UnlearnerApp
from pipeline.optuna_search_hp import instantiate_objects
from munl.datasets import get_loaders_from_dataset_and_unlearner_from_cfg_with_indices
from munl.configurations import get_img_size_for_dataset
from pathlib import Path
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlearner_name", type=str, default="naive")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--root", type=Path, default=".")
    parser.add_argument("--split_ndx", type=int, required=True)
    parser.add_argument("--forget_ndx", type=int, required=True)
    parser.add_argument("--model_seed", type=int, default=0)
    parser.add_argument("--data_seed", type=int, default=123)
    parser.add_argument("--overwrite", type=bool, default=False)
    return parser


def main(args):
    unlearner_name = args.unlearner_name
    model_name = args.model
    dataset_name = args.dataset
    root = args.root
    model_seed = args.model_seed
    data_seed = args.data_seed
    split_ndx = args.split_ndx
    forget_ndx = args.forget_ndx

    name_save_path = "artifacts/lira/unlearn"
    if not Path(name_save_path).exists():
        Path(name_save_path).mkdir(parents=True, exist_ok=True)
    img_size = get_img_size_for_dataset(dataset_name)
    dataset_cfg, unlearner, model_cfg = instantiate_objects(
        dataset_name=dataset_name, unlearner_name=unlearner_name, model_name=model_name
    )

    app = UnlearnerApp(
        model_cfg=model_cfg,
        dataset_cfg=dataset_cfg,
        unlearner=unlearner,
        model_seed=model_seed,
        random_state=data_seed,
    )
    save_path = Path(
        app.get_save_path(
            root=root,
            save=True,
            img_size=img_size,
            save_name=f"{split_ndx}_{forget_ndx}",
            save_path=name_save_path,
            unlearner_name=unlearner_name,
        )
    )
    print(f"Attempting to save at {save_path}")
    if save_path.exists() and not args.overwrite:
        print("Already exists, skipping.")
        return

    if unlearner_name in ["original", "naive"]:
        unlearner.cfg.num_epochs = 91

    retain, forget, val, test = get_retain_forget_val_test_indices(
        lira_path=root / "artifacts" / "lira" / "splits",
        split_ndx=split_ndx,
        forget_ndx=forget_ndx,
    )

    train_loader, retain_loader, forget_loader, val_loader, test_loader = (
        get_loaders_from_dataset_and_unlearner_from_cfg_with_indices(
            root=root,
            indices=[np.concatenate([retain, forget]), retain, forget, val, test],
            dataset_cfg=dataset_cfg,
            unlearner_cfg=unlearner.cfg,
        )
    )

    model = app.get_model(root / "artifacts" / dataset_name)
    original_model, unlearned_model = app.run_from_model_and_loaders(
        root=root,
        model=model,
        loaders=(train_loader, retain_loader, forget_loader, val_loader, test_loader),
        save=True,
        save_path=name_save_path,
        save_name=f"{split_ndx}_{forget_ndx}",
        unlearner_name=unlearner_name,
    )


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
