"""Surface Classification CNN Tuning Script.

This script performs tuning for a CNN model on surface classification task
given a set of input signals in order to evaluate its expected performance.
"""
import argparse
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from models.models import CNNSurfaceClassifier
from utils.datasets import CNNTrainingDataset
from utils.training import EarlyStopper, ProgressReporter, load_config, extract_experiment_name, set_seed, seed_worker, get_device, get_input_size, step
from utils.visualization import setup_matplotlib


def dataset_sufficiency_analysis(
        experiment_name,
        experiment_params,
        training_params,
        dataset_params,
        output_dir,
        progress_reporter=None
        ):
    """Perform tuning for CNN surface classification."""
    print(f"Running experiment: {experiment_name}")
    print(f"Subset: {experiment_params['channels']}, ",
          f"Configurations: {experiment_params['kinematics']}")

    g = set_seed(training_params['seed'])
    device = get_device()
    input_size = get_input_size(experiment_params['channels'])

    # Convert string paths to Path objects
    data_dir = Path(dataset_params['data_dir'])
    labels_file = Path(dataset_params['labels_file'])
    TRAIN_SIZES = np.arange(1, 9) / 10

    with open(labels_file, encoding='utf-8') as fp:
        labels = json.load(fp)

    dataset = [(data_dir.joinpath(key + '.csv'), values['surface'])
               for key, values in labels.items()
               if values['kinematics'] in experiment_params['kinematics']]

    x = [run[0] for run in dataset]
    target_classes = [run[1] for run in dataset]

    lb = LabelBinarizer()
    if dataset_params['generalized_classes']:
        generalized_classes = [
            'slippery' if label in ('3_Wykladzina_jasna', '4_Trawa')
            else 'grippy' if label in ('5_Spienione_PCV', '8_Pusta_plyta', '9_podklady', '10_Mata_ukladana')
            else 'neutral' for label in target_classes]
        lb.fit(generalized_classes)
        y = lb.transform(generalized_classes)
    else:
        lb.fit(target_classes)
        y = lb.transform(target_classes)
    classes = lb.classes_
    num_classes = len(classes)
    y = y.reshape(-1, num_classes)

    dataset_size = len(x)

    # Separate hold-out fold from dataset
    x_training, x_test, y_training, y_test = train_test_split(x, y, test_size=0.1, stratify=y)

    cv_training_data = CNNTrainingDataset(
        x_training,
        y_training,
        data_frequency=dataset_params['data_frequency'],
        window_length=200,
        subset=experiment_params['channels']
    )
    cv_test_data = CNNTrainingDataset(
        x_test,
        y_test,
        data_frequency=dataset_params['data_frequency'],
        window_length=200,
        subset=experiment_params['channels']
    )

    test_dataloader = DataLoader(
        cv_test_data,
        batch_size=training_params['batch_size'],
        worker_init_fn=seed_worker,
        generator=g
        )

    criterion = nn.CrossEntropyLoss()

    history = {}

    for train_size in TRAIN_SIZES:
        print(f"Train size: {train_size}")
        split_history = {}

        sss = StratifiedShuffleSplit(
            n_splits=40,
            test_size=int(0.1 * dataset_size),
            train_size=int(train_size * dataset_size)
            )
        n_splits = sss.get_n_splits()
        # Initialize progress reporter
        if progress_reporter:
            progress_reporter.set_total_steps(n_splits, training_params['num_epochs'])

        # Use simple progress bars since the launcher handles the main tracking
        fold_pbar = tqdm(total=n_splits, desc="CV Folds", disable=progress_reporter is not None)

        for i, (train_index, val_index) in enumerate(sss.split(x_training, y_training)):
            # Initialize the model in each split
            cnn_model = CNNSurfaceClassifier(input_size=input_size, output_size=num_classes).to(device)

            # Initialize optimizer in each split
            optimizer = torch.optim.Adam(
                cnn_model.parameters(),
                lr=1e-3,
                eps=1e-6,
                weight_decay=1e-3,
            )

            # Initialize scheduler in each split
            scheduler = ExponentialLR(optimizer, gamma=0.9)

            # Initialize early stopping
            early_stopper = EarlyStopper()

            train_dataloader = DataLoader(
                Subset(cv_training_data, train_index),
                batch_size=training_params['batch_size'],
                worker_init_fn=seed_worker,
                generator=g,
                shuffle=True,
            )
            val_dataloader = DataLoader(
                Subset(cv_training_data, val_index),
                batch_size=training_params['batch_size'],
                worker_init_fn=seed_worker,
                generator=g,
            )

            epoch_pbar = tqdm(total=training_params['num_epochs'],
                              desc=f"Fold {i+1}/{n_splits}",
                              disable=progress_reporter is not None)

            for epoch in range(training_params['num_epochs']):
                running_train_loss = 0.0
                running_val_loss = 0.0

                cnn_model.train()
                for idx, train_batch in enumerate(train_dataloader):
                    train_loss, _ = step(
                        cnn_model,
                        train_batch,
                        criterion,
                        device,
                        train=True,
                        optimizer=optimizer
                        )
                    running_train_loss += train_loss

                avg_train_loss = running_train_loss.detach().cpu().item() / (idx + 1)
                scheduler.step()

                cnn_model.eval()
                with torch.no_grad():
                    for idx, val_batch in enumerate(val_dataloader):
                        val_loss, _ = step(cnn_model, val_batch, criterion, device)
                        running_val_loss += val_loss

                avg_val_loss = running_val_loss.detach().cpu().item() / (idx + 1)

                if early_stopper.early_stop(avg_val_loss):
                    break
                if early_stopper.counter == 0:
                    best_model = cnn_model.state_dict()

                # Update progress
                if progress_reporter:
                    progress_reporter.update_epoch(epoch + 1, avg_train_loss, avg_val_loss)
                else:
                    epoch_pbar.set_postfix({
                        'Train Loss': f"{avg_train_loss:.2E}",
                        'Val Loss': f"{avg_val_loss:.2E}",
                    })
                    epoch_pbar.update(1)

                epoch_pbar.close()

            cnn_model.load_state_dict(best_model)

            y_true_train, y_pred_train = [], []
            running_train_loss = 0.0

            cnn_model.eval()
            with torch.no_grad():
                for idx, batch_train in enumerate(train_dataloader):
                    train_loss, train_outputs = step(
                                                    cnn_model,
                                                    batch_train,
                                                    criterion,
                                                    device,
                                                    )
                    running_train_loss += train_loss

                    y_true_train.extend(torch.argmax(batch_train[1], dim=1).cpu().numpy())
                    y_pred_train.extend(torch.argmax(train_outputs, dim=1).cpu().numpy())

            y_true_test, y_pred_test = [], []
            running_test_loss = 0.0

            cnn_model.eval()
            with torch.no_grad():
                for idx, test_batch in enumerate(test_dataloader):
                    test_loss, test_outputs = step(cnn_model, test_batch, criterion, device)
                    running_test_loss += test_loss

                    y_true_test.extend(torch.argmax(test_batch[1], dim=1).cpu().numpy())
                    y_pred_test.extend(torch.argmax(test_outputs, dim=1).cpu().numpy())

            if not progress_reporter:
                fold_pbar.update(1)

            split_history[i + 1] = {
                'train_f1_score': f1_score(y_true_train, y_pred_train, average='weighted'),
                'test_f1_score': f1_score(y_true_test, y_pred_test, average='weighted'),
            }
        history[train_size] = split_history

        fold_pbar.close()
        base_filename = '_'.join(['cnn', str(num_classes)] + experiment_params['kinematics'] + experiment_params['channels'])
        with open(output_dir / f'{base_filename}.json', 'w', encoding='utf-8') as fp:
            json.dump(history, fp)

        if progress_reporter:
            progress_reporter.set_completed()


def plot_results(history_filename, history_dir, output_dir=None):
    """Plot results from history file."""
    # set figure params
    TRAIN_SIZES = np.arange(1, 9) / 10

    # read data
    with open(history_dir / f'{history_filename}.json', encoding='utf-8') as fp:
        history = json.load(fp)

    # Set up matplotlib
    setup_matplotlib()

    # plot results
    train_curve, test_curve = [], []
    for scores_dict in history.values():
        train_f1_score, test_f1_score = [], []
        for split in scores_dict.values():
            train_f1_score.append(split['train_f1_score'])
            test_f1_score.append(split['test_f1_score'])
        train_curve.append(np.mean(train_f1_score[20:30]))
        test_curve.append(np.mean(test_f1_score[20:30]))
    plt.plot(TRAIN_SIZES, train_curve, label='train')
    plt.plot(TRAIN_SIZES, test_curve, label='test')
    plt.yticks(ticks=np.arange(4, 11) / 10)
    plt.yticks(ticks=np.arange(45, 100, 5) / 100, minor=True)
    plt.xlim(0.1, 0.8)
    plt.ylim(0.4, 1)
    plt.xlabel('train set size')
    plt.ylabel('F1 score')
    plt.grid(which='major', axis='both', linewidth=1)
    plt.grid(which='minor', axis='y', linewidth=0.4)
    plt.minorticks_on()
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_dir.joinpath('pessimistic_biases.png'), dpi=300, bbox_inches="tight")


def main():
    """Main script for loading configuration file and running the experiment."""
    parser = argparse.ArgumentParser(description="CNN Cross-Validation for Surface Classification")
    parser.add_argument(
        '--config-file',
        type=str,
        help="YAML configuration file name"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help="Output directory path",
        default='results/logs/biases'
    )
    parser.add_argument(
        '--progress-dir',
        type=Path,
        help="Progress directory path (optional)",
        default=None
    )
    args = parser.parse_args()

    config_path = Path('configs').joinpath(args.config_file)
    all_params = load_config(config_path)
    experiments, training_params, dataset_params = all_params.values()
    experiment_params = next((params['experiment_params'] for params in experiments))
    experiment_name = extract_experiment_name(experiment_params)

    # Setup progress reporter if progress directory is provided
    progress_reporter = None
    if args.progress_dir:
        progress_file = Path(args.progress_dir) / f'{args.experiment_name}_progress.json'
        progress_reporter = ProgressReporter(progress_file)

    generalized_classes = '3' if dataset_params['generalized_classes'] else '10'

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # dataset_sufficiency_analysis(
    #     experiment_name,
    #     experiment_params,
    #     training_params,
    #     dataset_params,
    #     output_dir,
    #     progress_reporter
    #     )
    # time.sleep(1)

    plot_results(
        history_filename='_'.join(['cnn', generalized_classes] + experiment_params['kinematics'] + experiment_params['channels']),
        history_dir=output_dir,
        output_dir=Path('results/figures')
        )


if __name__ == '__main__':
    main()
