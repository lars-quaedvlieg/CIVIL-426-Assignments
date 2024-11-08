import argparse
from functools import reduce
from operator import mul

import numpy as np
import optuna
import torch

from src.main import set_seed, Trainer


def evaluate_model(params, model_train_func):
    n_runs = 5  # Number of runs to average over, you can decrease this number to speed up the optimization
    all_target_accuracies = []
    for i in range(n_runs):
        seed = params["seed"] + i
        set_seed(seed)
        final_target_acc = model_train_func()
        all_target_accuracies.append(final_target_acc)
    return np.mean(all_target_accuracies)

def objective_baseline(trial, default_params, trial_cat_params):
    params = default_params.copy()
    for param_name, param_values in trial_cat_params.items():
        params[param_name] = trial.suggest_categorical(param_name, param_values)
        params[param_name] = trial.suggest_categorical(param_name, param_values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(argparse.Namespace(**params), device)

    target_acc = evaluate_model(params, trainer.train_baseline)
    return target_acc

def objective_coral(trial, default_params, trial_cat_params):
    params = default_params.copy()
    for param_name, param_values in trial_cat_params.items():
        params[param_name] = trial.suggest_categorical(param_name, param_values)
        params[param_name] = trial.suggest_categorical(param_name, param_values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(argparse.Namespace(**params), device)

    target_acc = evaluate_model(params, trainer.train_coral)
    return target_acc

def objective_adversarial(trial, default_params, trial_cat_params):
    params = default_params.copy()
    for param_name, param_values in trial_cat_params.items():
        params[param_name] = trial.suggest_categorical(param_name, param_values)
        params[param_name] = trial.suggest_categorical(param_name, param_values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(argparse.Namespace(**params), device)

    target_acc = evaluate_model(params, trainer.train_adversarial)
    return target_acc

def objective_adabn(trial, default_params, trial_cat_params):
    params = default_params.copy()
    for param_name, param_values in trial_cat_params.items():
        params[param_name] = trial.suggest_categorical(param_name, param_values)
        params[param_name] = trial.suggest_categorical(param_name, param_values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(argparse.Namespace(**params), device)

    target_acc = evaluate_model(params, trainer.train_adabn)
    return target_acc

def run_study(study_name, objective_fn, search_space, folder):
    study = optuna.create_study(storage=f'sqlite:///{folder}/study.db',
                                study_name=study_name,
                                direction="maximize",
                                load_if_exists=True,
                                sampler=optuna.samplers.GridSampler(search_space)
                                )
    study.optimize(objective_fn, n_trials=reduce(mul, [len(val_arr) for val_arr in search_space.values()], 1))

    # Visualize the importance of hyperparameters
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()

    # # Load the study for resuming, comment out when reloading
    # study = optuna.load_study(study_name=study_name, storage=f'sqlite:///{folder}/study.db')
    # ### Visualizing impact of hyperparameters

    # Contour plot between two hyperparameters
    # fig = optuna.visualization.plot_contour(study, params=['base_lr', 'n_layer'])
    # fig.show()

    # you can get a dataframe of the hyperparamter results with:
    # hyper_df = study.trials_dataframe()
    # print(hyper_df.head())

def main():
    folder = 'out'
    default_params = {
        'batch_size': 32,
        'lr': 0.0001,
        'epochs': 100,
        'seed': 42,
        'coral_weight': 1.0,
        'adversarial_weight': 1.0,
    }

    # Baseline
    study_name = 'hypertune_baseline'
    search_space = {
        'lr': [0.00001, 0.0005, 0.0001, 0.001],
        'batch_size': [4, 8, 16, 32, 64],
    }
    objective_fn = lambda trial: objective_baseline(trial, default_params, search_space)
    run_study(study_name, objective_fn, search_space, folder)

    # Coral
    study_name = 'hypertune_coral'
    search_space = {
        'lr': [0.00001, 0.0005, 0.0001, 0.001],
        'batch_size': [4, 8, 16, 32, 64],
        'coral_weight': [0.1, 0.5, 1, 2, 4],
    }
    objective_fn = lambda trial: objective_coral(trial, default_params, search_space)
    run_study(study_name, objective_fn, search_space, folder)

    # Adversarial
    study_name = 'hypertune_adversarial'
    search_space = {
        'lr': [0.00001, 0.0005, 0.0001, 0.001],
        'batch_size': [4, 8, 16, 32, 64],
        'adversarial_weight': [0.1, 0.5, 1, 2, 4],
    }
    objective_fn = lambda trial: objective_adversarial(trial, default_params, search_space)
    run_study(study_name, objective_fn, search_space, folder)

    # Adabn
    study_name = 'hypertune_adabn'
    search_space = {
        'lr': [0.00001, 0.0005, 0.0001, 0.001],
        'batch_size': [4, 8, 16, 32, 64],
    }
    objective_fn = lambda trial: objective_adabn(trial, default_params, search_space)
    run_study(study_name, objective_fn, search_space, folder)

if __name__ == '__main__':
    main()