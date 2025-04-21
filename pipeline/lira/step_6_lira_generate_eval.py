def main():
    with open("commands/all_lira_eval.txt", "w") as out_fo:
        for unlearner in ["kgltop4", "kgltop5", "finetune", "kgltop6", "kgltop3", "kgltop2", "original", "naive"]:
            for split_ndx in range(64):
                for forget_ndx in range(10):
                    out_fo.write(
                        f"python munl/lira/gather_lira_predictions.py --unlearner={unlearner} --split_ndx={split_ndx} --forget_ndx={forget_ndx}\n"
                    )


if __name__ == "__main__":
    main()