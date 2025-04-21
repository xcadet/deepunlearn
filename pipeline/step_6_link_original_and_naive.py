#!/usr/bin/env python

import argparse
from pathlib import Path

import yaml

from munl.utils import create_or_update_symlinks


def get_parser():
    parser = argparse.ArgumentParser(
        description="Combine the contents of the original runs"
    )
    parser.add_argument(
        "--original_runs_yaml",
        type=str,
        default="commands/original_runs.yaml",
        help="Path to the original runs",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="artifacts",
        help="Path to the artifacts directory",
    )
    parser.add_argument("--no_strict", action="store_true")
    return parser


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(args):
    strict = not args.no_strict
    datasets = load_yaml(Path(args.original_runs_yaml))
    artifacts_dir = Path(args.artifacts_dir)
    for dataset in datasets:
        assert len(dataset) == 1
        dataset_name, unlearners = list(dataset.items())[0]
        assert len(unlearners) == 2
        for unlearner_info in unlearners:
            unlearner, required_dirs = list(unlearner_info.items())[0]
            unlearners_path = artifacts_dir / dataset_name / "unlearn"
            original_dir_path = unlearners_path / f"unlearner_{unlearner}"
            if not original_dir_path.exists():
                original_dir_path.mkdir(parents=True)
            expected_only = []
            for required_dir in required_dirs:
                required_dir_path = unlearners_path / required_dir
                if not required_dir_path.exists() and not strict:
                    continue
                print(f"Trying to bind {required_dir_path} into {original_dir_path}")
                create_or_update_symlinks(
                    source_dir=required_dir_path,
                    target_dir=original_dir_path,
                )
                expected_only.extend(
                    list(map(lambda path: path.name, required_dir_path.iterdir()))
                )
            expected_only = set(expected_only)
            in_original = set(
                list(map(lambda path: path.name, original_dir_path.iterdir()))
            )
            if strict:
                missing = expected_only.difference(in_original)
                print(missing)
                assert len(missing) == 0, "There should only be a perfect match"


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
