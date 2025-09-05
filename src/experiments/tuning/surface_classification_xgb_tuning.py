"""Surface Classification CNN Cross-Validation Script.

This script performs cross-validation for training a CNN model on surface classification task
given a set of input signals in order to evaluate its expected performance.
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Subset
from xgboost import XGBClassifier
from utils.datasets import XGBTrainingDataset
from utils.training import load_config, extract_experiment_name, set_seed, seed_worker, average_over_splits


THRESHOLD = 5e-3


def xgb_training(
        experiment_name,
        experiment_params,
        training_params,
        dataset_params,
        output_dir,
        ):
    """Perform cross-validation for XGBoost surface classification."""
    print(f"Running experiment: {experiment_name}")
    print(f"Subset: {experiment_params['channels']}, ",
          f"Configurations: {experiment_params['kinematics']}")

    g = set_seed(training_params['seed'])

    # Convert string paths to Path objects
    data_dir = Path(dataset_params['data_dir'])
    labels_file = Path(dataset_params['labels_file'])

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

    history = {}
    history_features = {}

    sss = StratifiedShuffleSplit(n_splits=40, test_size=0.2)

    for i, (training_index, test_index) in enumerate(sss.split(x, y)):
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
            n_jobs=4,
            verbose=10,
        )
        clf_search.fit(x_train, y_train)

        # Find the most important features from the best estimator
        importances = clf_search.best_estimator_.feature_importances_
        idx = np.arange(len(importances))

        # Take features above threshold.
        best_features = idx[importances > THRESHOLD]

        # Fit the estimator on the best sets of hyperparameters and features
        xgb_tuned = XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
            **clf_search.best_params_,
        )
        xgb_tuned.fit(x_train[:, best_features], y_train)
        y_pred = xgb_tuned.predict(x_test[:, best_features])

        history[i + 1] = classification_report(y_true, y_pred, output_dict=True)
        history_features[i + 1] = {feature_name: feature_importance for feature_name, feature_importance in zip(cv_data.engineered_features[best_features].tolist(), importances[best_features].tolist())}

    base_filename = '_'.join(['xgb', str(num_classes)] + experiment_params['kinematics'] + experiment_params['channels'])
    with open(output_dir / f'{base_filename}.json', 'w', encoding='utf-8') as fp:
        json.dump(history, fp)
    with open(output_dir / f'{base_filename}_features.json', 'w', encoding='utf-8') as fp:
        json.dump(history_features, fp)

    # Average classification reports over splits and save to JSON files
    average_over_splits(history, base_filename, output_dir)


def main():
    """Main script for loading configuration file and running the experiment."""
    parser = argparse.ArgumentParser(description="CNN Cross-Validation for Surface Classification")
    parser.add_argument(
        '--config-file',
        type=str,
        help="YAML configuration file name"
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        help="Experiment ID to run"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help="Output directory path"
    )
    args = parser.parse_args()

    config_path = Path('configs').joinpath(args.config_file)
    all_params = load_config(config_path)
    experiments, training_params, dataset_params = all_params.values()
    experiment_name = args.experiment_name
    experiment_params = next((params['experiment_params'] for params in experiments if extract_experiment_name(params['experiment_params']) == experiment_name))

    generalized_classes = '3' if dataset_params['generalized_classes'] else '10'

    output_dir = args.output_dir.joinpath(generalized_classes, '_'.join(experiment_params['kinematics']))
    output_dir.mkdir(parents=True, exist_ok=True)

    xgb_training(
        experiment_name,
        experiment_params,
        training_params,
        dataset_params,
        output_dir,
        )
    time.sleep(1)


if __name__ == "__main__":
    main()
