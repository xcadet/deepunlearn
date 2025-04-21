
def generate_all_optuna(
    output_path,
    objectives,
    datasets,
    models,
    unlearners,
    model_seeds,
    split_indices,
    forget_indices,
    num_trials,
):
    with open(output_path, "w") as out_fo:
        for objective in objectives:
            for dataset in datasets:
                for model in models:
                    for unlearner in unlearners:
                        for model_seed in model_seeds:
                            for split_ndx in split_indices:
                                for forget_ndx in forget_indices:
                                    out_fo.write(
                                        f"python pipeline/optuna_search_hp.py --dataset={dataset} --unlearner={unlearner} --model={model} --model_seed={model_seed} --objective={objective} --lira --split_ndx={split_ndx} --forget_ndx={forget_ndx} --num_trials={num_trials}\n"
                                    )


if __name__ == "__main__":
    datasets = ["cifar10"]
    model_seeds = [0]
    models = ["resnet18"]
    split_indices = list(range(32))
    forget_indices = list(range(1))
    num_trials = 200
    unlearners = ["kgltop2", "kgltop3", "kgltop4", "kgltop5", "kgltop6", "finetune"]
    generate_all_optuna(
        "commands/lira_optuna.txt",
        ["objective10"],
        datasets,
        models,
        unlearners,
        model_seeds,
        split_indices=split_indices,
        forget_indices=forget_indices,
        num_trials=num_trials,
    )
