from pipeline import DATASETS, MODEL_SEEDS, MODELS, OBJECTIVES, UNLEARNERS
import argparse
from pathlib import Path
from munl.utils import extract_list_of_strings


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objectives",
        type=extract_list_of_strings,
        default=OBJECTIVES,
        help="Objectives to use",
    )
    parser.add_argument(
        "--datasets",
        type=extract_list_of_strings,
        default=DATASETS,
        help="Datasets to use",
    )
    parser.add_argument(
        "--models",
        type=extract_list_of_strings,
        default=MODELS,
        help="Models to use",
    )
    parser.add_argument(
        "--unlearners",
        type=extract_list_of_strings,
        default=UNLEARNERS,
        help="Unlearners to use",
    )
 
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("commands/all_optuna.txt"),
        help="Path to the output file",
    )
   
    return parser


def generate_all_optuna(
    output_path: Path, objectives: list[str], datasets: list[str], models: list[str], unlearners: list[str], model_seeds: list[int]
):
    with open(output_path, "w") as out_fo:
        for objective in objectives:
            for dataset in datasets:
                for model in models:
                    for unlearner in unlearners:
                        for model_seed in model_seeds:
                            out_fo.write(
                                f"python pipeline/optuna_search_hp.py --dataset={dataset} --unlearner={unlearner} --model={model} --model_seed={model_seed} --objective={objective}\n"
                            )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    output_path = Path(args.output_path)
    generate_all_optuna(
        args.output_path, OBJECTIVES, DATASETS, MODELS, UNLEARNERS, MODEL_SEEDS
    )
