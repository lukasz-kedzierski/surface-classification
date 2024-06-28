import json
import numpy as np
import pandas as pd
import random
import time

from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset

from datasets import SurfaceDataset
from helpers import EarlyStopper, step
from models import CNNSurfaceClassifier

device = "cuda:0" if torch.cuda.is_available() else "cpu"

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

BATCH_SIZE = 32
INPUT_SIZE = 2
NUM_EPOCHS = 100
DATA_DIR = Path('../data/train_set/csv/')
HISTORY_DIR = Path('../results/tuning/')
LOOKBACK = 8/3
SAMPLING_FREQUENCY = 75.
DATASET_FREQUENCY = 150.
SUBSET = ('servo',)
CONFIGURATIONS = ('6W',)

with open('../data/train_set/labels.json') as fp:
    labels = json.load(fp)

dataset = [(DATA_DIR.joinpath(key + '.csv'), values['surface'])
           for key, values in labels.items()
           if values['kinematics'] in CONFIGURATIONS and values['spacing'] == 'R1' and 'T1' in values['trajectory']]

X = pd.Series([run[0] for run in dataset], name='bag_name')
y_primary = [run[1] for run in dataset]

y_secondary = []
# y_secondary = ['slippery' if label in ('1_Panele', '5_Spienione_PCV', '6_Linoleum')
#                else 'grippy' if label in ('3_Wykladzina_jasna', '8_Pusta_plyta', '9_podklady')
#                else 'neutral' for label in y_primary]
# y_secondary = ['slippery' if label in ('3_Wykladzina_jasna', '4_Trawa')
#                else 'grippy' if label in ('5_Spienione_PCV', '8_Pusta_plyta', '9_podklady', '10_Mata_ukladana')
#                else 'neutral' for label in y_primary] # Pawel set
# y_secondary = ['slippery' if label in ('3_Wykladzina_jasna', '4_Trawa')
#                else 'grippy' if label in ('2_Wykladzina_czarna', '5_Spienione_PCV', '9_podklady', '10_Mata_ukladana')
#                else 'neutral' for label in y_primary] # Clustering set

lb = LabelBinarizer()
if y_secondary:
    lb.fit(y_secondary)
    y = lb.transform(y_secondary)
else:
    lb.fit(y_primary)
    y = lb.transform(y_primary)
classes = lb.classes_
num_classes = len(classes)
y = y.reshape(-1, num_classes)

cv_data = SurfaceDataset(
    X,
    y,
    sample_freq=SAMPLING_FREQUENCY,
    data_freq=DATASET_FREQUENCY,
    lookback=LOOKBACK,
    subset=SUBSET,
)

criterion = nn.CrossEntropyLoss()

history = {}

sss = StratifiedShuffleSplit(test_size=0.2)
for i, (training_index, test_index) in enumerate(sss.split(X, y)):
    # Initialize the model in each split
    cnn_model = CNNSurfaceClassifier(input_size=INPUT_SIZE, output_size=num_classes).to(device)
    best_model = cnn_model.state_dict()
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

    # Separate hold-out fold
    train_index, val_index = train_test_split(training_index, test_size=0.2, stratify=y[training_index])

    train_dataloader = DataLoader(
        Subset(cv_data, train_index),
        batch_size=BATCH_SIZE,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        Subset(cv_data, val_index),
        batch_size=BATCH_SIZE,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_dataloader = DataLoader(
        Subset(cv_data, test_index),
        batch_size=BATCH_SIZE,
        worker_init_fn=seed_worker,
        generator=g,
    )

    train_batches = len(train_dataloader)
    val_batches = len(val_dataloader)

    for epoch in range(NUM_EPOCHS):
        running_train_loss = 0.0
        running_val_loss = 0.0

        pbar = tqdm(train_dataloader, total=train_batches)
        cnn_model.train()
        for idx, train_batch in enumerate(pbar):
            train_loss, _ = step(cnn_model, train_batch, criterion, device, train=True, optimizer=optimizer)
            running_train_loss += train_loss

            pbar.set_description(
                f"Fold {i + 1}/{sss.get_n_splits()},"
                f"Epoch {epoch + 1}/{NUM_EPOCHS},"
                f"Training loss: {running_train_loss / (idx + 1):.2E}")
        scheduler.step()

        pbar_val = tqdm(val_dataloader, total=val_batches)
        cnn_model.eval()
        with torch.no_grad():
            for idx, val_batch in enumerate(pbar_val):
                val_loss, _ = step(cnn_model, val_batch, criterion, device)
                running_val_loss += val_loss

                pbar_val.set_description(
                    f"Fold {i + 1}/{sss.get_n_splits()},"
                    f"Epoch {epoch + 1}/{NUM_EPOCHS},"
                    f"Validation loss: {running_val_loss / (idx + 1):.2E}")

        validation_loss = running_val_loss / (idx + 1)
        if early_stopper.early_stop(validation_loss):
            print(f"Split {i + 1} ended on epoch {epoch + 1 - early_stopper.patience}!")
            break
        if early_stopper.counter == 0:
            best_model = cnn_model.state_dict()

    cnn_model.load_state_dict(best_model)

    test_batches = len(test_dataloader)
    y_true, y_pred = [], []
    running_test_loss = 0.0

    pbar_test = tqdm(test_dataloader, total=test_batches)
    cnn_model.eval()
    with torch.no_grad():
        for idx, test_batch in enumerate(pbar_test):
            test_loss, test_outputs = step(cnn_model, test_batch, criterion, device)
            running_test_loss += test_loss

            y_true.extend(torch.argmax(test_batch[1], dim=1).cpu().numpy())
            y_pred.extend(torch.argmax(test_outputs, dim=1).cpu().numpy())

            pbar_test.set_description(
                f"Fold {i + 1}/{sss.get_n_splits()}, Test loss: {running_test_loss / (idx + 1):.2E}")

    history[i + 1] = {'accuracy': accuracy_score(y_true, y_pred), 'f1_score': f1_score(y_true, y_pred, average='macro')}

history_filename = '_'.join((str(num_classes),) + CONFIGURATIONS + SUBSET) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S")
json.dump(history, open(HISTORY_DIR / f'{history_filename}.json', 'w'))
