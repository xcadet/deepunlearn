from pathlib import Path
from sqlite3 import OperationalError
from typing import Optional

import optuna
from pandas import DataFrame as DataFrame

from munl.hpsearch.utils import sqlify
from pipeline.optuna_search_hp import format_optuna_save_dir, format_study_name


def load_trials_from_path(path: Path) -> DataFrame:
    database_path = sqlify(path)
    try:
        storage = optuna.storages.RDBStorage(database_path)
    except OperationalError as e:
        print(f"Error: {e} on '{path}'")
        raise e
    except Exception as e:
        print(f"Error: {e} on '{path}'")
        raise e

    study_summaries = optuna.study.get_all_study_summaries(storage=storage)
    for study_summary in study_summaries:
        study_name = study_summary.study_name

    loaded_study = optuna.load_study(study_name=study_name, storage=database_path)
    df = loaded_study.trials_dataframe()
    df = df.dropna(subset=[col for col in df.columns if "values_" in col])
    only_complete = df[(df.state == "COMPLETE")]
    return only_complete


def get_trials(
    dataset: str,
    unlearner: str,
    model: str,
    model_seed: int,
    objective: str,
    num_trials: int = 100,
    split_ndx: Optional[int] = None,
    forget_ndx: Optional[int] = None,
) -> DataFrame:
    """Extract trials from stored under the optuna search.

    Args:
        dataset (str): Name of the dataset. (e.g. cifar10)
        unlearner (str): Name of the unlearning method. (e.g. finetune)
        model_seed (int): Seed of the model. (e.g. 0)
        root (_type_, optional): Root to start the search from. Defaults to ROOT.
        num_trials (int, optional): Number of trials ran. Defaults to 100.

    Returns:
        (DataFrame): All trias from the study
    """
    assert model in ["vit11m", "resnet18"], f"Model '{model}' not supported."
    study_name = format_study_name(
        dataset_name=dataset,
        unlearner_name=unlearner,
        model_name=model,
        model_seed=model_seed,
        objective=objective,
        split_ndx=split_ndx,
        forget_ndx=forget_ndx,
    )
    sub_dir = "optuna"
    if split_ndx is not None:
        sub_dir = "lira/optuna"
    path = format_optuna_save_dir(dataset, study_name, subdir=sub_dir) / "study.db"

    if not path.exists():
        print(f"Skipping '{path}' (Not found.)")
        return None
    return load_trials_from_path(path).head(num_trials)


def get_best_trials(trials, top_k: int = 1):
    if trials is None:
        return None
    new = trials.copy()

    # Drop rows with NaN values only in score columns
    score_columns = [col for col in new.columns if "values_" in col]
    if len(score_columns) == 0:
        assert "value" in new.columns
        score_columns = ["value"]

    new = new.dropna(subset=score_columns)

    def calculate_euclidean(row):
        return sum(row[col] ** 2 for col in score_columns) ** 0.5

    new["score"] = new.apply(calculate_euclidean, axis=1)

    sorted_trials = new.sort_values("score", ascending=True)
    best_trial = sorted_trials.head(top_k)


    best_score = best_trial["score"].values[0]
    assert best_score == new["score"].min()

    return best_trial

def generate_pt_list(
    output_file,
    root,
    datasets,
    methods,
    model_seeds=[0, 1, 2],
    num_trials=100,
):
    needed = []
    dataset_to_unlearner_to_seed_to_path = {}
    for dataset in datasets:
        dataset_to_unlearner_to_seed_to_path[dataset] = {}
        for unlearner in methods:
            dataset_to_unlearner_to_seed_to_path[dataset][unlearner] = {}
            for model_seed in model_seeds:
                path = f"{dataset}/optuna/opt-{dataset}-{unlearner}-{model_seed}-{num_trials}"
                best = get_best_trials(
                    get_trials(
                        dataset, unlearner, model_seed, num_trials=num_trials, root=root
                    )
                )
                if best is None:
                    continue
                best_number = best["number"].values[0]
                target_path = str(Path("artifacts") / path / f"{best_number}.pt")
                dataset_to_unlearner_to_seed_to_path[dataset][unlearner][model_seed] = (
                    best_number,
                    target_path,
                )
                needed.append(target_path)
    with open(output_file, "w") as out_fo:
        for line in needed:
            out_fo.write(line + "\n")

    return dataset_to_unlearner_to_seed_to_path
