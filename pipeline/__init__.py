DATASETS = ["cifar10", "cifar100", "mnist", "fashion_mnist", "utkface"]
MODELS = ["resnet18", "vit11m"]
MODEL_SEEDS = [0, 1, 2]
ALL_MODEL_SEEDS = range(10)
REFERENCES = ["original", "naive"]
UNLEARNERS = [
    "kgltop1",
    "kgltop2",
    "kgltop3",
    "kgltop4",
    "kgltop5",
    "kgltop6",
    "kgltop7",
    "finetune",
    "bad_teacher",
    "successive_random_labels",
    "gradient_ascent",
    "neggradplus",
    "fisher",
    "salun",
    "scrubv2",
    "influence",
    "cfk",
    "euk",
]
ALL_UNLEARNERS = UNLEARNERS + REFERENCES
OBJECTIVES = ["objective10"]
