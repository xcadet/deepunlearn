import argparse
from pathlib import Path
from typing import List, Optional, Union

from pandas import DataFrame

from munl.hpsearch.find_trials import get_best_trials, get_trials


def parse_ints_as_list(string: str) -> List[int]:
    return list(map(int, string.split(",")))


def parse_path(string: str) -> Path:
    return Path(string)


def get_parser():
    parser = argparse.ArgumentParser(description="Find best hyper parameters")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--unlearner", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--objective", type=str, required=True)
    parser.add_argument("--model_seeds", type=parse_ints_as_list, default=[0, 1, 2])
    parser.add_argument(
        "--output_path", type=parse_path, default=Path("artifacts/hyper_parameters.txt")
    )
    parser.add_argument("--append", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--lira", action="store_true", default=False)
    parser.add_argument("--split_ndx", type=parse_ints_as_list, default=None)
    parser.add_argument("--forget_ndx", type=parse_ints_as_list, default=None)
    return parser


def get_all_best_trials(
    dataset: str,
    unlearner: str,
    model: str,
    model_seeds: List[int],
    objective: str,
    num_trials: int,
    split_ndx: Optional[List[int]],
    forget_ndx: Optional[List[int]],
) -> List[Union[DataFrame]]:
    best_trials: List[Union[DataFrame]] = []
    for split_id in split_ndx if split_ndx is not None else [None]:
        for forget_id in forget_ndx if forget_ndx is not None else [None]:
            for model_seed in model_seeds:
                trials = get_trials(
                    dataset,
                    unlearner,
                    model=model,
                    model_seed=model_seed,
                    objective=objective,
                    num_trials=num_trials,
                    split_ndx=split_id,
                    forget_ndx=forget_id,
                )
                if trials is None or len(trials) < num_trials:
                    tup = (dataset, unlearner, model, model_seed, objective)
                    print(
                        f"There is a missing trial {tup} "
                        + ("0" if trials is None else f"{len(trials)}")
                        + f"/{num_trials}"
                    )
                    best_trials.append(None)
                    continue
                best_trial = get_best_trials(trials)
                assert len(best_trial) == 1
                best_trials.append(best_trial.iloc[0])
    return best_trials


def ends_with_any(string: str, suffixes: List[str]) -> bool:
    return any(string.endswith(suffix) for suffix in suffixes)


class FindBestHyperParametersObtained:
    def __init__(self, lira: bool):
        self.lira = lira
        self.cmd = self._get_cmd_from_lira()

    def _get_cmd_from_lira(self):
        return (
            "python pipeline/lira/step_1_unlearn_lira.py "
            if self.lira
            else "python pipeline/step_3_unlearn.py "
        )

    def run(
        self,
        dataset: str,
        unlearner: str,
        model: str,
        model_seeds: List[int],
        objective: str,
        output_path: Path,
        num_trials: int,
        split_ndx: Optional[List[int]] = None,
        forget_ndx: Optional[List[int]] = None,
        append: bool = True,
    ):
        save_path = ("lira_" if self.lira else "") + objective
        trials = get_all_best_trials(
            dataset,
            unlearner,
            model,
            model_seeds,
            objective,
            num_trials,
            split_ndx=split_ndx,
            forget_ndx=forget_ndx,
        )
        if any(trial is None for trial in trials):
            print("Some trials are missing for", dataset, unlearner, model, objective)
            return
        trials.sort(key=lambda trial: trial["score"])
        overal_best_trial = trials[0]
        hyper_params = list(
            filter(lambda name: name.startswith("params_"), overal_best_trial.index)
        )
        loadable = [f"dataset={dataset}", f"unlearner={unlearner}", f"model={model}"]
        for name in hyper_params:
            value = overal_best_trial[name]
            cfg_name = name.replace("params_", "unlearner.cfg.")
            if cfg_name.endswith("kd_t") or cfg_name.endswith("sga_num_epochs"):
                continue
            if ends_with_any(
                cfg_name,
                [
                    "batch_size",
                    "num_epochs",
                    "num_blocks",
                    "msteps",
                    "training_epoch_factor",
                ],
            ):
                value = int(value)
            complete = f"{cfg_name}={value}"
            loadable.append(complete)
        loadable.append(f"save_path={save_path}")
        res = self.cmd + " ".join(loadable)
        if append is True:
            with open(output_path, "a") as out_fo:
                if split_ndx is None:
                    for model_seed in range(10):
                        res_seed = res + f" model_seed={model_seed}"
                        out_fo.write(res_seed + "\n")
                else:
                    for split_ndx in range(64):
                        for forget_ndx in range(10):
                            res_seed = (
                                res
                                + f" model_seed={0} split_ndx={split_ndx} forget_ndx={forget_ndx}"
                            )
                            out_fo.write(res_seed + "\n")


def main(args):
    assert not args.lira or (args.split_ndx is not None and args.forget_ndx is not None)
    app = FindBestHyperParametersObtained(args.lira)
    app.run(
        dataset=args.dataset,
        unlearner=args.unlearner,
        model=args.model,
        model_seeds=args.model_seeds,
        objective=args.objective,
        output_path=args.output_path,
        num_trials=args.num_trials,
        split_ndx=args.split_ndx,
        forget_ndx=args.forget_ndx,
        append=args.append,
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
