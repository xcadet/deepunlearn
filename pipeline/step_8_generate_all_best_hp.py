# The goal of this script is to generate all the optuna searches such that we can launch them via a slurm submit
from munl.hpsearch.find_best_hyper_parameters import (
    get_parser as hp_get_parser,
    main as hp_main,
)
from pipeline import DATASETS, MODELS, OBJECTIVES, UNLEARNERS
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
        default="commands/all_best_hp.txt",
        help="Path to save the output commands",
    )
    return parser


def generate_all_best_hp(output_path, objectives, datasets, models, unlearners):
    with open(output_path, "w") as _:
        pass
    for objective in objectives:
        for dataset in datasets:
            for model in models:
                for unlearner in unlearners:
                    local_args = hp_get_parser().parse_args(
                        [
                            "--dataset",
                            dataset,
                            "--unlearner",
                            unlearner,
                            "--model",
                            model,
                            "--objective",
                            objective,
                            "--output_path",
                            str(output_path),
                            "--append",
                        ]
                    )
                    hp_main(local_args)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    generate_all_best_hp(
        args.output_path, args.objectives, args.datasets, args.models, args.unlearners
    )
