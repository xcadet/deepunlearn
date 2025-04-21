from argparse import Namespace
import argparse


def main(args: Namespace):
    num_splits = args.num_splits
    num_forgets = args.num_forgets
    output_file = args.output_file
    unlearners = args.unlearners
    models = args.models
    with open(output_file, "w") as out_fo:
        for model in models:
            for unlearner in unlearners:
                for split_ndx in range(num_splits):
                    for forget_ndx in range(num_forgets):
                        out_fo.write(
                            f"python3 pipeline/lira/step_lira_run.py --model={model} --unlearner_name={unlearner} --split_ndx {split_ndx} --forget_ndx {forget_ndx}\n"
                        )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_splits", type=int, default=64)
    parser.add_argument("--num_forgets", type=int, default=10)
    parser.add_argument("--unlearners", type=list, default=["naive", "original"])
    parser.add_argument("--models", type=list, default=["resnet18"])
    parser.add_argument(
        "--output_file",
        type=str,
        default="commands/lira_original_and_naive_train_instructions.txt",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
