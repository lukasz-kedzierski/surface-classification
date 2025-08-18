"""Module with helper objects used in training and inference of surface prediction models."""


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
