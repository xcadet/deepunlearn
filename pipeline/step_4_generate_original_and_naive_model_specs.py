import argparse
import itertools
from pathlib import Path

import yaml
import os


def get_hydra_cluster() -> str:
    return os.environ.get("HYDRA_CLUSTER_LAUNCHER", "")


def load_yaml(path: Path):
    data = yaml.safe_load(path.read_text())
    return data


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs_yaml", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser


def generate_configurations(config_options):
    processed_options = {}
    for item in config_options:
        for key, value in item.items():
            # Check if value is a comma-separated string and split it
            if isinstance(value, str) and "," in value:
                value = [v.strip() for v in value.split(",")]
            elif not isinstance(value, list):  # Ensure all values are lists
                value = [value]
            processed_options[key] = value

    # Generate all combinations of configuration options
    combinations = []
    for values in itertools.product(*processed_options.values()):
        config_string = " ".join(
            f"{k}={v}" for k, v in zip(processed_options.keys(), values)
        )
        combinations.append(config_string)

    return combinations


def process_dataset(dataset_name, config_options, output_fo):
    # Generate configurations
    configurations = generate_configurations(config_options)

    for config in configurations:
        complete_config = (
            f"python pipeline/step_5_unlearn.py {config} dataset={dataset_name} "
            + get_hydra_cluster()
        )
        output_fo.write(complete_config + "\n")


def process_yaml_datasets(datasets, output_fo):
    # Process each dataset
    for dataset in datasets:
        for dataset_name, config_options in dataset.items():
            process_dataset(dataset_name, config_options, output_fo)


def main(args):
    datasets = load_yaml(Path(args.specs_yaml))
    with open(args.output_path, "w") as out_fo:
        process_yaml_datasets(datasets, output_fo=out_fo)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
