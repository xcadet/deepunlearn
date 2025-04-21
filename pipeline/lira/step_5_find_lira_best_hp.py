from munl.hpsearch.find_best_hyper_parameters import (
    get_parser as hp_get_parser,
    main as hp_main,
)
import argparse
from pathlib import Path
from munl.utils import extract_list_of_ints, extract_list_of_strings


def get_parser():
    parser = argparse.ArgumentParser(description="Find best hyper parameters")
    parser.add_argument(
        "--datasets",
        type=extract_list_of_strings,
        default=["cifar10"],
        help="List of datasets to run on",
    )
    parser.add_argument(
        "--unlearners",
        type=extract_list_of_strings,
        default=["kgltop4", "kgltop5", "finetune", "kgltop6", "kgltop3", "kgltop2"],
        help="List of unlearners to run on",
    )
    parser.add_argument(
        "--models",
        type=extract_list_of_strings,
        default=["resnet18"],
        help="List of models to run on",
    )
    parser.add_argument(
        "--objectives",
        type=extract_list_of_strings,
        default=["objective10"],
        help="List of objectives to run on",
    )
    parser.add_argument(
        "--split_indices",
        type=extract_list_of_ints,
        default=list(range(32)),
        help="List of split indices to run on",
    )
    parser.add_argument(
        "--forget_indices",
        type=extract_list_of_ints,
        default=list(range(1)),
        help="List of forget indices to run on",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default="commands/lira_best_hp.txt",
        help="Path to save the output commands",
    )
    return parser


def generate_all_best_hp(
    output_path: Path,
    objectives: list[str],
    datasets: list[str],
    models: list[str],
    unlearners: list[str],
    split_indices: list[int],
    forget_indices: list[int],
):
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
                            "--model_seed",
                            "0",
                            "--append",
                            "--lira",
                            "--split_ndx",
                            ",".join(map(str, split_indices)),
                            "--forget_ndx",
                            ",".join(map(str, forget_indices)),
                        ]
                    )
                    hp_main(local_args)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    generate_all_best_hp(
        args.output_path,
        args.objectives,
        args.datasets,
        args.models,
        args.unlearners,
        args.split_indices,
        args.forget_indices,
    )
