# Deep Unlearn: Benchmarking Machine Unlearning for Image Classification

This repository is the official implementation of [Deep Unlearn: Benchmarking Machine Unlearning for Image Classification](https://arxiv.org/abs/2410.01276)

## Abstract

Machine unlearning (MU) aims to remove the influence of particular data points from the learnable parameters of a trained machine learning model. This is a crucial capability in light of data privacy requirements, trustworthiness, and safety in deployed models. MU is particularly challenging for deep neural networks (DNNs), such as convolutional nets or vision transformers, as such DNNs tend to memorize a notable portion of their training dataset. Nevertheless, the community lacks a rigorous and multifaceted study that looks into the success of MU methods for DNNs. In this paper, we investigate 18 state-of-the-art MU methods across various benchmark datasets and models, with each evaluation conducted over 10 different initializations, a comprehensive evaluation involving MU over 100K models. We show that, with the proper hyperparameters, Masked Small Gradients (MSG) and Convolution Transpose (CT), consistently perform better in terms of model accuracy and run-time efficiency across different models, datasets, and initializations, assessed by population-based membership inference attacks (MIA) and per-sample unlearning likelihood ratio attacks (U-LiRA). Furthermore, our benchmark highlights the fact that comparing a MU method only with commonly used baselines, such as Gradient Ascent (GA) or Successive Random Relabeling (SRL), is inadequate, and we need better baselines like Negative Gradient Plus (NG+) with proper hyperparameter selection.

## Requirements
We recommend using conda to install the requirements.

```setup
conda env create -f environment/munl.yaml
conda activate munl
pip install -e .
```

We gathered the different steps into the `pipeline/` directory.

## 1. Preparing the splits
We use [hydra](https://hydra.cc/docs/intro/) to run parts of the pipeline.
To prepare the dataset splits run the following command:

```splits
python pipeline/step_1_generate_fixed_splits.py dataset=mnist,fashion_mnist,cifar10,cifar100 --multirun
```

For [UTKFace](https://susanqq.github.io/UTKFace/), we use the dataset's `Aligned&Cropped Faces` version.

```splits
python pipeline/step_1_generate_fixed_splits.py dataset=utkface --multirun
```

## 2. Generate the initial models
To generate the untrained models, use the following commands:

For the ResNet18 models:
```resnet18_initial
 python pipeline/step_2_generate_model_initialization.py model=resnet18 num_classes=2,5,10,100 model_seed="range(10)" --multirun
```

For the TinyViT:
```vit_initial
 python pipeline/step_2_generate_model_initialization.py model=vit11m img_size=32,64,224 num_classes=2,5,10,100 model_seed="range(10)" --multirun
```
## 3. Link the initial models to their associated datasets folders
To avoid creating initial models for each dataset we link the different initial models to the datasets:
```link
bash pipeline/step_3_create_and_link_model_initializations_dir.sh 
```
## 4. Generate Original and Naive Model instructions

```link
python pipeline/step_4_generate_original_and_naive_model_specs.py --specs_yaml pipeline/datasets_original_and_naive_hyper.yaml --output_path commands/train_original_and_naive_instructions.txt
```
This generates `./commands/train_original_and_naive_instructions.txt`

## 5. Run the Original and Naive training phase
Each line of `./commands/train_original_and_naive.txt` is a command that can be invoked as is.

For instance, one can run `python pipeline/step_5_unlearn.py unlearner=original unlearner.cfg.num_epochs=50 unlearner.cfg.batch_size=256 unlearner.cfg.optimizer.learning_rate=0.1 model=resnet18 model_seed=0 dataset=mnist`
Executing all these lines will train the original and naive models for all 5 datasets.
These models then serve as starting point (original) and reference (retrained) for the next steps.

## 6. Linking the original and naive models
Then we link the original and retrained models so that they can be used in the next steps.

```
python pipeline/step_6_link_original_and_naive.py
```

## 6. Hyperparameter search
Once we have the original and retrained models, we can proceed to the hyperparameter search.
The original models serve as starting point to the unlearning method.
While the retrained models are evaluate the performance of the models.
```
python pipeline/step_7_generate_optuna.py
```
This generates `commands/all_optuna.txt`

## 7. Run the different searches
Each line of `./commands/all_optuna.txt` is a command that can be invoked as is.
Similarly to step 5, each line of the file can be called separately.

## 8. Extract the best hyperparameter per unlearning method
Once the search are complete, one can run the following:

```
pipeline/step_8_generate_all_best_hp.py
```
This generates `commands/all_best_hp.txt`, which follows a similar format to step 5 and 7.

## 9. Unlearn using the best hyperparameter
Calling the different lines of `commands/all_best_hp.txt`, run the unlearning methods with the best set of hyperparameter found.
Once these models are unlearned they are ready for evaluation.

## U-LIRA: We separated the pipeline for the U-LIRA case into `pipeline/lira`
### 1. Generating the U-LIRA splits:
First we need to generate the splits to run the lira evaluation.

```lira_splits
python pipeline/lira/step_1_lira_generate_splits.py
```

### 2. Generating the instruction to train the original and naive models:
```lira_original_naive
python pipeline/lira/step_2_lira_generate_original_and_naive_instructions.py
```
This generates `./commands/lira_original_and_naive_train_instructions.txt`

### 3. Running the Original and Retrained training phase
Each line of `./commands/lira_original_and_naive_train_instructions.txt` is a command that can be invoked as is.
Executing all these lines will train the original and retrained models for the lira experiments.
These models then serve as starting point (original) and reference (retrained) for the next steps.

### 4. Generating the optuna searches
```lira_optuna
python pipeline/lira/step_4_lira_generate_optuna.py
```
This will generate `./commands/lira_optuna.txt`, containing the commands to run the hyper-parameter searchers for the LIRA pipeline.

### 5. Running the searches
This will generate `./commands/lira_optuna.txt`
Similarly to the main pipeline steps 5, 7 and 9, each line can be processed independently and will perform the hyper-parameter search.

### 6. Finding the best hyper-parameters:
Once the different searches are complete, one can run the following to search for the best combination of hyperparameter for the unlearning methods to evaluate.
```python
pipeline/lira/step_5_find_lira_best_hp.py
```
This will create `commands/lira_best_hp.txt` where each line will train an unlearning method with the best hyperparameter found during the search.

### 7. Preparing the evaluation
Once the unlearned models are generated, one can run the following:
```
pipeline/lira/step_6_lira_generate_eval.py
```
Which creates `commands/all_lira_eval.txt`

### 8. Extracting the predictions of the different models
Each line of `commands/all_lira_eval.txt` will proceed to extract the predictions of the unlearned models across the entire dataset.

### 9. Generating the membership tests
Once the different model predictions are extracted, the following script generates `commands/membership_commands.txt`

```python
pipeline/lira/step_7_generate_membership_commands.py
```

Each line of `commands/membership_commands.txt` produces the `lira/<unlearning_method>_membership.npy`
Which contains for each data point from the forget set its membership scores.


##  Citation
If you found our work useful please consider citing it:

```bibtex
@misc{cadet2024deepunlearnbenchmarkingmachine,
      title={Deep Unlearn: Benchmarking Machine Unlearning}, 
      author={Xavier F. Cadet and Anastasia Borovykh and Mohammad Malekzadeh and Sara Ahmadi-Abhari and Hamed Haddadi},
      year={2024},
      url={https://arxiv.org/abs/2410.01276}, 
}
```

## Acknowledgments
We would like to express our gratitude to all references in our paper that open-sourced their codebase, methodology, and dataset, which served as the foundation for our work.