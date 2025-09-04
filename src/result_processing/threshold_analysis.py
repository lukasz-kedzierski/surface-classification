"""Surface Classification CNN Cross-Validation Script.

This script performs cross-validation for training a CNN model on surface classification task
given a set of input signals in order to evaluate its expected performance.
"""

import json
import time
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Subset
from xgboost import XGBClassifier
from utils.datasets import XGBTrainingDataset
from utils.training import load_config, set_seed, seed_worker


def xgb_threshold_analysis(
        experiment_name,
        experiment_params,
        training_params,
        dataset_params,
        ):
    """Perform cross-validation for XGBoost threshold analysis."""

    print(f"Running experiment: {experiment_name}")
    print(f"Subset: {experiment_params['channels']}, ",
          f"Configurations: {experiment_params['kinematics']}")

    g = set_seed(training_params['seed'])

    # Convert string paths to Path objects
    data_dir = Path(dataset_params['data_dir'])
    labels_file = Path(dataset_params['labels_file'])

    image_dir = Path('results/figures')
    image_dir.mkdir(parents=True, exist_ok=True)

    with open(labels_file, encoding='utf-8') as fp:
        labels = json.load(fp)

    dataset = [(data_dir.joinpath(key + '.csv'), values['surface'])
               for key, values in labels.items()
               if values['kinematics'] in experiment_params['kinematics']]

    x = [run[0] for run in dataset]
    target_classes = [run[1] for run in dataset]

    le = LabelEncoder()
    if dataset_params['generalized_classes']:
        generalized_classes = [
            'slippery' if label in ('3_Wykladzina_jasna', '4_Trawa')
            else 'grippy' if label in ('5_Spienione_PCV', '8_Pusta_plyta', '9_podklady', '10_Mata_ukladana')
            else 'neutral' for label in target_classes]
        le.fit(generalized_classes)
        y = le.transform(generalized_classes)
    else:
        le.fit(target_classes)
        y = le.transform(target_classes)
    classes = le.classes_
    num_classes = len(classes)

    cv_data = XGBTrainingDataset(
        x,
        y,
        data_frequency=dataset_params['data_frequency'],
        window_length=200,
        subset=experiment_params['channels'],
        time_features=dataset_params['time_features'],
        freq_features=dataset_params['freq_features']
        )

    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 4, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0.1, 0.5],
        'reg_lambda': [0.1, 0.5],
    }

    t_95 = 2.228
    nicer_blue = '#00A0FF'

    thresholds = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    threshold_history = defaultdict(list)

    sss = StratifiedShuffleSplit(test_size=0.2)

    for (training_index, test_index) in sss.split(x, y):
        # Initialize the model in each split
        xgb_model = XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
        )

        train_subset = Subset(cv_data, training_index)
        test_subset = Subset(cv_data, test_index)

        train_dataloader = DataLoader(
            train_subset,
            batch_size=len(train_subset),
            worker_init_fn=seed_worker,
            generator=g,
            shuffle=True,
        )
        test_dataloader = DataLoader(
            test_subset,
            batch_size=len(test_subset),
            worker_init_fn=seed_worker,
            generator=g,
        )

        # Extract the whole datasets to variables
        x_train, y_train = next(iter(train_dataloader))
        x_test, y_true = next(iter(test_dataloader))

        # Find the best hyperparameters
        clf_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=8,
            verbose=10,
        )
        clf_search.fit(x_train, y_train)

        # Find the most important features from the best estimator
        importances = clf_search.best_estimator_.feature_importances_
        idx = np.arange(len(importances))  # indexes with the highest importance

        # Test different thresholds
        for threshold in thresholds:

            best_features = idx[importances > threshold]

            # Fit the estimator on the best sets of hyperparameters and features
            xgb_tuned = XGBClassifier(
                objective='multi:softprob',
                num_class=num_classes,
                **clf_search.best_params_,
            )
            xgb_tuned.fit(x_train[:, best_features], y_train)
            y_pred = xgb_tuned.predict(x_test[:, best_features])

            f1 = f1_score(y_true, y_pred, average='weighted')
            threshold_history[threshold].append(f1)

    # Calculate average F1-scores and standard deviations
    average_f1 = []
    ci_f1 = []

    for threshold in thresholds:
        scores = threshold_history[threshold]
        average_f1.append(np.mean(scores))
        ci_f1.append(t_95 * np.std(scores) / np.sqrt(len(scores)))

    # Find optimal threshold
    optimal_idx = np.argmax(average_f1)
    optimal_threshold = thresholds[optimal_idx]

    plt.figure(figsize=(12, 3))
    plt.errorbar(thresholds, average_f1, yerr=ci_f1,
                 marker='o', capsize=4, capthick=1, linewidth=1, markersize=4, c='k')

    plt.axvline(optimal_threshold, c=nicer_blue)

    plt.xlabel('importance threshold')
    plt.ylabel('average F1-score')
    plt.title(f"Optimal threshold: {optimal_threshold} (f1-score = {average_f1[optimal_idx]:.4f})")
    plt.grid(True)
    plt.xscale("log")
    plt.xlim([5e-06, 1e-01])
    plt.tight_layout()
    plt.savefig(image_dir.joinpath('threshold_analysis.png'), dpi=300)


def main():
    """Main script for loading configuration file and running the experiment."""

    all_params = load_config(Path('configs/threshold_analysis.yaml'))
    experiment_name, experiment_params, training_params, dataset_params = all_params.values()

    xgb_threshold_analysis(
        experiment_name,
        experiment_params,
        training_params,
        dataset_params,
        )
    time.sleep(1)


if __name__ == "__main__":
    main()
