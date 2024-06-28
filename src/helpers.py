class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def step(model, batch, criterion, device, train=False, optimizer=None):
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
