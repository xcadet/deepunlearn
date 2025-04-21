import argparse
import copy
from pathlib import Path

import omegaconf
from pipeline.optuna_utils import get_loaders
import gc
import time

import optuna
import torch
from hydra.utils import instantiate
from typing import Optional

from munl.configurations import (
    dataset_store,
    get_img_size_for_dataset,
    model_store,
    unlearner_store,
)
from munl.models import get_model_from_cfg
from munl.hpsearch.objectives import unlearner_optuna
from munl.hpsearch.suggestor import HyperParameterSuggestor
from munl.settings import DEFAULT_BATCH_SIZE, DEFAULT_OPTUNA_N_TRIALS
from pipeline.step_5_unlearn import UnlearnerApp

OBJECTIVE_TO_FUNC = {
    "objective10": unlearner_optuna,
}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--unlearner", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_seed", type=int, default=1)
    parser.add_argument("--random_state", type=int, default=123)
    parser.add_argument("--subdir", type=str, default="optuna")
    parser.add_argument("--lira", action="store_true")
    parser.add_argument("--split_ndx", type=int, default=0)
    parser.add_argument("--forget_ndx", type=int, default=0)
    parser.add_argument(
        "--objective",
        type=str,
        choices=OBJECTIVE_TO_FUNC.keys(),
    )
    parser.add_argument("--num_trials", type=int, default=100)
    return parser


def get_configuration(dataset_name, unlearner_name, model_name):
    dataset_conf = dataset_store["dataset", dataset_name]
    model_conf = model_store["model", model_name]
    unlearner_conf = unlearner_store["unlearner", unlearner_name]
    return dataset_conf, unlearner_conf, model_conf


def instantiate_objects(dataset_name: str, model_name: str, unlearner_name: str):
    dataset_conf, unlearner_conf, model_conf = get_configuration(
        dataset_name=dataset_name, model_name=model_name, unlearner_name=unlearner_name
    )

    dataset_instance = instantiate(dataset_conf) if dataset_conf else None
    model_instance = instantiate(model_conf) if model_conf else None
    unlearner_instance = instantiate(unlearner_conf) if unlearner_conf else None

    return dataset_instance, unlearner_instance, model_instance


def to_optimize(
    trial,
    args,
    save_dir,
    objective_function,
    lira: bool,
    split_ndx: Optional[int],
    forget_ndx: Optional[int],
):
    print("split_ndx", split_ndx, "forget_ndx", forget_ndx)
    save_name = None
    assert (split_ndx is not None and forget_ndx is not None) or (
        split_ndx is None and forget_ndx is None
    )
    if lira:
        assert split_ndx is not None and forget_ndx is not None
        save_name = f"{split_ndx}_{forget_ndx}"

    dataset_name = args.dataset
    unlearner_name = args.unlearner
    model_name = args.model
    print("Starting", model_name)
    dataset_cfg, unlearner, model_cfg = instantiate_objects(
        dataset_name=dataset_name,
        model_name=model_name,
        unlearner_name=unlearner_name,
    )
    model_seed = args.model_seed
    random_state = args.random_state
    img_size = get_img_size_for_dataset(dataset_name)
    root = Path(".")
    naive_cfg = copy.deepcopy(unlearner.cfg)
    original_cfg = copy.deepcopy(unlearner.cfg)
    if lira:
        naive_cfg.model_initializations_dir = "unlearn/naive"
        original_cfg.model_initializations_dir = "unlearn/original"
        weights_root = root / "artifacts" / dataset_name / "artifacts" / "lira"
        name_save_path = root / "artifacts" / dataset_name / "lira" / "unlearn"
    else:
        naive_cfg.model_initializations_dir = "unlearn/unlearner_naive"
        original_cfg.model_initializations_dir = "unlearn/unlearner_original"
        weights_root = root / "artifacts" / dataset_name
        name_save_path = root / "artifacts" / dataset_name / "unlearn"

    naive_model = get_model_from_cfg(
        root=weights_root,
        model_cfg=model_cfg,
        unlearner_cfg=naive_cfg,
        num_classes=dataset_cfg.num_classes,
        model_seed=model_seed,
        img_size=img_size,
        split_ndx=split_ndx,
        forget_ndx=forget_ndx,
    )
    model = get_model_from_cfg(
        root=weights_root,
        model_cfg=model_cfg,
        unlearner_cfg=original_cfg,
        num_classes=dataset_cfg.num_classes,
        model_seed=model_seed,
        img_size=img_size,
        split_ndx=split_ndx,
        forget_ndx=forget_ndx,
    )

    hyper_parameters = unlearner.HYPER_PARAMETERS
    suggestor = HyperParameterSuggestor(dataset_name, lira=lira)
    suggestor.suggest_in_place(unlearner.cfg, hyper_parameters, trial)
    if dataset_name == "utkface" and unlearner.cfg.batch_size > 128:
        print("Pruning trial due to batch_size.")
        raise optuna.exceptions.TrialPruned()
    print(f"Unlearner Configuration {unlearner}")

    app = UnlearnerApp(
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        unlearner=unlearner,
        model_seed=model_seed,
        random_state=random_state,
    )
    ###
    img_size = get_img_size_for_dataset(dataset_name)
    save_name = None
    train_loader, retain_loader, forget_loader, val_loader, test_loader = get_loaders(
        root=root,
        dataset_cfg=dataset_cfg,
        unlearner_cfg=unlearner.cfg,
        lira=lira,
        split_ndx=split_ndx,
        forget_ndx=forget_ndx,
        random_state=random_state,
    )
    original_model, unlearned_model = app.run_from_model_and_loaders(
        root=root,
        model=model,
        loaders=[
            train_loader,
            retain_loader,
            forget_loader,
            val_loader,
            test_loader,
        ],
        save=True,
        save_path=name_save_path,
        save_name=save_name,
        unlearner_name=unlearner_name,
    )
    from munl.settings import default_loaders
    unlearner.cfg.loaders = omegaconf.DictConfig(default_loaders())
    train_loader, retain_loader, forget_loader, val_loader, test_loader = get_loaders(
        root=root,
        dataset_cfg=dataset_cfg,
        unlearner_cfg=unlearner.cfg,
        lira=lira,
        split_ndx=split_ndx,
        forget_ndx=forget_ndx,
        random_state=random_state,
    )

    torch.cuda.empty_cache()
    gc.collect()
    save_path = save_dir / f"{trial.number}.pt"
    print(f"Saved model at path {save_path}")
    torch.save(unlearned_model.state_dict(), save_path)
    return objective_function(
        original_model,
        naive_model,
        unlearned_model,
        dataset_name,
        DEFAULT_BATCH_SIZE if dataset_name != "utkface" else 32,
        random_state,
        retain_loader,
        forget_loader,
        val_loader,
        test_loader,
    )


def format_study_name(
    dataset_name: str,
    unlearner_name: str,
    model_name: str,
    model_seed: int,
    objective: str,
    split_ndx: Optional[int] = None,
    forget_ndx: Optional[int] = None,
) -> str:
    lira_part = (
        f"-{split_ndx}_{forget_ndx}"
        if split_ndx is not None and forget_ndx is not None
        else ""
    )
    name = f"opt-{dataset_name}-{unlearner_name}"
    name += f"-{model_name}-{model_seed}{lira_part}-{objective}"
    return name


def format_optuna_save_dir(
    dataset_name: str, study_name: str, subdir: str = "optuna"
) -> Path:
    save_dir = Path("artifacts") / f"{dataset_name}" / subdir / f"{study_name}"
    return save_dir


def get_study_and_save_dir(
    dataset,
    model_seed,
    study_name,
    objective,
    subdir="optuna",
    objective_to_func=OBJECTIVE_TO_FUNC,
):
    save_dir = format_optuna_save_dir(dataset, study_name, subdir=subdir)
    save_dir.mkdir(parents=True, exist_ok=True)
    db_path = save_dir / "study.db"
    storage_url = "sqlite:///" + str(db_path)

    if objective in [
        "objective1",
        "objective2",
        "objective3",
        "objective6",
        "objective7",
        "objective8",
    ]:
        sampler = optuna.samplers.NSGAIIISampler(seed=model_seed)
        directions = ["minimize", "minimize", "minimize"]
    elif objective in ["objective4"]:
        sampler = optuna.samplers.TPESampler(seed=model_seed)
        directions = ["minimize"]
    elif objective in ["objective5", "objective9", "objective10"]:
        sampler = optuna.samplers.NSGAIIISampler(seed=model_seed)
        directions = ["minimize", "minimize", "minimize", "minimize"]
    else:
        raise NotImplementedError(f"Objective '{objective}' not implemented.")

    study = optuna.create_study(
        study_name=study_name,
        directions=directions,
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True,
    )

    return study, save_dir, objective_to_func[objective]


class OptunaSearchApp:
    def __init__(
        self,
        subdir: str = "optuna",
        lira: bool = False,
        split_ndx: int = 0,
        forget_ndx: int = 0,
        num_trials: int = DEFAULT_OPTUNA_N_TRIALS,
    ):
        # Handle LiRA as as speical case
        self.subdir = ("lira/" if lira else "") + subdir
        self.num_trials = num_trials
        self.lira = lira
        self.split_ndx = split_ndx
        self.forget_ndx = forget_ndx

    def run(
        self, dataset: str, unlearner: str, model: str, model_seed: int, objective: str
    ):
        num_trials = self.num_trials
        subdir = self.subdir
        study_name = format_study_name(
            dataset,
            unlearner,
            model,
            model_seed,
            objective=objective,
            split_ndx=self.split_ndx,
            forget_ndx=self.forget_ndx,
        )
        study, save_dir, objective_func = get_study_and_save_dir(
            dataset=dataset,
            model_seed=model_seed,
            study_name=study_name,
            objective=objective,
            subdir=subdir,
        )
        print(f"Starting study '{study_name}' for {num_trials} trials in '{save_dir}'.")
        # NOTE: Old version considered failed trial, we need only completed trials
        # num_trials_already_run = len(study.trials)
        retries_left = 3
        previous_complete = -1
        while True:
            num_trials_already_run = len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            )
            num_trials_to_run = min(num_trials, num_trials - num_trials_already_run)

            msg = f"Study '{study_name}' has already completed "
            msg += f"{num_trials_already_run} trials. "
            msg += f"And a total of {len(study.trials)} trials. "
            msg += f"Running up to {num_trials_to_run} more trials."
            print(msg)

            if num_trials_to_run > 0 and retries_left > 0:
                study.optimize(
                    lambda trial: to_optimize(
                        trial,
                        args,
                        save_dir,
                        objective_function=objective_func,
                        lira=self.lira,
                        split_ndx=self.split_ndx,
                        forget_ndx=self.forget_ndx,
                    ),
                    # n_trials=num_trials_to_run,
                    n_trials=1,
                )
                if len(study.trials) == previous_complete:
                    retries_left -= 1
                else:
                    retries_left = 3
            else:
                msg = "No more trials are needed; the study has "
                msg += "reached the maximum number of trials."
                print(msg)
                break


# Would greatly benefit from having the sqlite save system instead of pickled results
def main(args):
    lira_mode = args.lira
    split_ndx = args.split_ndx
    forget_ndx = args.forget_ndx
    print(f"LiRA mode: {lira_mode}, split_ndx: {split_ndx}, forget_ndx: {forget_ndx}")
    if not lira_mode:
        print("Overriding split_ndx and forget_ndx to None.")
        split_ndx = None
        forget_ndx = None
    print(f"LiRA mode: {lira_mode}, split_ndx: {split_ndx}, forget_ndx: {forget_ndx}")

    subdir = args.subdir
    app = OptunaSearchApp(
        subdir=subdir,
        lira=lira_mode,
        split_ndx=split_ndx,
        forget_ndx=forget_ndx,
        num_trials=args.num_trials,
    )
    app.run(
        dataset=args.dataset,
        unlearner=args.unlearner,
        model=args.model,
        model_seed=args.model_seed,
        objective=args.objective,
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
