"""Surface Classification CNN Cross-Validation Script.

This script performs cross-validation for training a CNN model on surface classification task
given a set of input signals in order to evaluate its expected performance.
"""

import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from datasets import CNNTrainingDataset
from models import CNNSurfaceClassifier
from utils.training import step


class ProgressReporter:
    """Report progress to file for monitoring by launcher."""

    def __init__(self, progress_file):
        self.progress_file = Path(progress_file)
        self.progress_data = {
            'total_steps': 0,
            'current_step': 0,
            'status': 'initializing',
            'current_fold': 0,
            'total_folds': 0,
            'current_epoch': 0,
            'total_epochs': 0,
            'train_loss': 0.0,
            'val_loss': 0.0
        }
        self._write_progress()

    def _write_progress(self):
        """Write current progress to file."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress_data, f)
        except Exception:
            # If we can't write progress, don't crash the training
            pass

    def set_total_steps(self, total_folds, total_epochs):
        """Set total steps for progress tracking."""
        self.progress_data['total_steps'] = total_folds * total_epochs
        self.progress_data['total_folds'] = total_folds
        self.progress_data['total_epochs'] = total_epochs
        self.progress_data['status'] = 'training'
        self._write_progress()

    def update_fold(self, current_fold):
        """Update current fold."""
        self.progress_data['current_fold'] = current_fold
        self._write_progress()

    def update_epoch(self, current_epoch, train_loss=None, val_loss=None):
        """Update current epoch and losses."""
        self.progress_data['current_epoch'] = current_epoch
        self.progress_data['current_step'] = ((self.progress_data['current_fold'] - 1) *
                                             self.progress_data['total_epochs'] + current_epoch)

        if train_loss is not None:
            self.progress_data['train_loss'] = float(train_loss)
        if val_loss is not None:
            self.progress_data['val_loss'] = float(val_loss)

        self._write_progress()

    def set_completed(self):
        """Mark as completed."""
        self.progress_data['status'] = 'completed'
        self.progress_data['current_step'] = self.progress_data['total_steps']
        self._write_progress()

    def set_failed(self):
        """Mark as failed."""
        self.progress_data['status'] = 'failed'
        self._write_progress()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id):
    """Set the seed for worker processes to ensure reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device():
    """Get the device to run the model on."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def get_input_size(channels):
    """Get the input size based on the subset of modalities."""
    if channels is None:
        return 0
    size_map = {'imu': 6, 'servo': 2}
    return sum(size_map[input] for input in channels)


def load_config(config_file):
    """Load configuration from YAML file."""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def cnn_cv(
        experiment_name,
        experiment_params,
        training_params,
        dataset_params,
        output_dir,
        progress_reporter=None
        ):
    """Perform cross-validation for CNN surface classification."""
    print(f"Running experiment: {experiment_name}")
    print(f"Subset: {experiment_params['channels']}, ",
          f"Configurations: {experiment_params['kinematics']}")

    g = set_seed(training_params['seed'])
    device = get_device()
    input_size = get_input_size(experiment_params['channels'])

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

    cv_data = CNNTrainingDataset(
        x,
        y,
        data_frequency=dataset_params['data_frequency'],
        window_length=200,
        subset=experiment_params['channels']
        )

    criterion = nn.CrossEntropyLoss()

    history = {}

    sss = StratifiedShuffleSplit(test_size=0.2)
    n_splits = sss.get_n_splits()
    # Initialize progress reporter
    if progress_reporter:
        progress_reporter.set_total_steps(n_splits, training_params['num_epochs'])

    # Use simple progress bars since the launcher handles the main tracking
    fold_pbar = tqdm(total=n_splits, desc="CV Folds", disable=progress_reporter is not None)

    for i, (train_index, val_index) in enumerate(sss.split(x, y)):
        if progress_reporter:
            progress_reporter.update_fold(i + 1)

        split_train_loss = []
        split_val_loss = []
        split_accuracy = []

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

        train_dataloader = DataLoader(
            Subset(cv_data, train_index),
            batch_size=training_params['batch_size'],
            worker_init_fn=seed_worker,
            generator=g,
            shuffle=True,
        )
        val_dataloader = DataLoader(
            Subset(cv_data, val_index),
            batch_size=training_params['batch_size'],
            worker_init_fn=seed_worker,
            generator=g,
        )

        epoch_pbar = tqdm(total=training_params['num_epochs'],
                            desc=f"Fold {i+1}/{n_splits}",
                            disable=progress_reporter is not None)

        for epoch in range(training_params['num_epochs']):
            idx = 0
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
            split_train_loss.append(avg_train_loss)
            scheduler.step()

            y_true, y_pred = [], []

            cnn_model.eval()
            with torch.no_grad():
                for idx, val_batch in enumerate(val_dataloader):
                    val_loss, val_outputs = step(cnn_model, val_batch, criterion, device)
                    running_val_loss += val_loss

                    y_true.extend(torch.argmax(val_batch[1], dim=1).cpu().numpy())
                    y_pred.extend(torch.argmax(val_outputs, dim=1).cpu().numpy())

                avg_val_loss = running_val_loss.detach().cpu().item() / (idx + 1)
                split_val_loss.append(avg_val_loss)

            split_accuracy.append(f1_score(y_true, y_pred, average='weighted'))

            # Update progress
            if progress_reporter:
                progress_reporter.update_epoch(epoch + 1, avg_train_loss, avg_val_loss)
            else:
                epoch_pbar.set_postfix({
                    'Train Loss': f"{avg_train_loss:.2E}",
                    'Val Loss': f"{avg_val_loss:.2E}",
                    'F1': f"{split_accuracy[-1]:.3f}"
                })
                epoch_pbar.update(1)

        epoch_pbar.close()
        if not progress_reporter:
            fold_pbar.update(1)

        history[i + 1] = {
            'train_loss': split_train_loss,
            'val_loss': split_val_loss,
            'accuracy': split_accuracy
            }

    fold_pbar.close()

    base_filename = '_'.join(['cv', str(num_classes)] + experiment_params['kinematics'] + experiment_params['channels'])
    with open(output_dir / f'{base_filename}.json', 'w', encoding='utf-8') as fp:
        json.dump(history, fp)

    if progress_reporter:
        progress_reporter.set_completed()


def main():
    """Main script for loading configuration file and running the experiment."""
    parser = argparse.ArgumentParser(description="CNN Cross-Validation for Surface Classification")
    parser.add_argument(
        '--config',
        type=Path,
        help="YAML configuration file path"
    )
    parser.add_argument(
        '--experiment-id',
        type=str,
        help="Experiment ID to run"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help="Output directory path"
    )
    parser.add_argument(
        '--progress-dir',
        type=Path,
        help="Progress directory path (optional)",
        default=None
    )
    args = parser.parse_args()

    # Setup progress reporter if progress directory is provided
    progress_reporter = None
    if args.progress_dir:
        progress_file = Path(args.progress_dir) / f"{args.experiment_id}_progress.json"
        progress_reporter = ProgressReporter(progress_file)

    all_params = load_config(args.config)
    experiments, training_params, dataset_params = all_params.values()
    experiment_name = args.experiment_id
    experiment_params = next((params['experiment_params'] for params in experiments if params['experiment_name'] == experiment_name))

    cnn_cv(experiment_name, experiment_params, training_params, dataset_params, args.output_dir, progress_reporter)
    time.sleep(1)


if __name__ == "__main__":
    main()
