"""Module with helper objects used in training and inference of surface prediction models."""
import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml


class EarlyStopper:
    """Helper class for early stopping during training of neural networks."""
    def __init__(self, patience=10, min_delta=1e-5):
        """
        Parameters
        ----------
        patience : int, default=10
            Number of epochs with no improvement after which training will be stopped.
        min_delta : float, default=1e-5
            Minimum change in the monitored quantity to qualify as an improvement.

        Attributes
        ----------
        counter : int
            Counter for epochs without improvement.
        min_validation_loss : float
            Minimum validation loss observed so far.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        """Check if training should be stopped based on validation loss.

        Parameters
        ----------
        validation_loss : float
            Current validation loss to compare with the minimum observed loss.

        Returns
        -------
        bool
            True if training should be stopped, False otherwise.
        """
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


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
            with open(self.progress_file, 'w', encoding='utf-8') as f:
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


def load_config(config_file):
    """Load configuration from YAML file."""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


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


def step(model, batch, criterion, device, train=False, optimizer=None):
    """Perform a single training or validation step.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to train or validate.
    batch : tuple
        A tuple containing input features and target labels.
    criterion : torch.nn.Module
        Loss function to compute the loss.
    device : torch.device
        Device to perform computations on (CPU or GPU).
    train : bool, default=False
        If True, perform a training step; if False, perform a validation step.
    optimizer : torch.optim.Optimizer, optional
        Optimizer for updating model parameters during training.

    Returns
    -------
    tuple
        Loss value and model outputs.
    """
    batch_x, batch_y = batch
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    batch_x = batch_x.permute(0, 2, 1)
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)

    if train:
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss, outputs
