import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time

from cycler import cycler
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset

from datasets import CNNTrainingDataset
from helpers import EarlyStopper
from models import CNNSurfaceClassifier

# # set device
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
#
# # set seed
# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
#
#
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)
#
#
# g = torch.Generator()
# g.manual_seed(0)
#
# # set variables
# BATCH_SIZE = 32
# INPUT_SIZE = 6
# NUM_EPOCHS = 100
# DATA_DIR = Path('../data/train_set/csv/')
# HISTORY_DIR = Path('../results/biases/')
# LOOKBACK = 8/3
# SAMPLING_FREQUENCY = 75.
# DATASET_FREQUENCY = 150.
# SUBSET = ('imu',)
# CONFIGURATIONS = ('4W',)
TRAIN_SIZES = np.arange(1, 9) / 10
#
# # load and split data
# with open('../data/train_set/labels.json') as fp:
#     labels = json.load(fp)
# dataset = [(DATA_DIR.joinpath(key + '.csv'), values['surface'])
#            for key, values in labels.items()
#            if values['kinematics'] in CONFIGURATIONS and values['spacing'] == 'R1' and 'T1' in values['trajectory']]
# X = pd.Series([run[0] for run in dataset], name='bag_name')
# y_primary = [run[1] for run in dataset]
# y_secondary = []
# # y_secondary = ['slippery' if label in ('1_Panele', '5_Spienione_PCV', '6_Linoleum')
# #                else 'grippy' if label in ('3_Wykladzina_jasna', '8_Pusta_plyta', '9_podklady')
# #                else 'neutral' for label in y_primary]
# # y_secondary = ['slippery' if label in ('3_Wykladzina_jasna', '4_Trawa')
# #                else 'grippy' if label in ('5_Spienione_PCV', '8_Pusta_plyta', '9_podklady', '10_Mata_ukladana')
# #                else 'neutral' for label in y_primary] # Pawel set
# # y_secondary = ['slippery' if label in ('3_Wykladzina_jasna', '4_Trawa')
# #                else 'grippy' if label in ('2_Wykladzina_czarna', '5_Spienione_PCV', '9_podklady', '10_Mata_ukladana')
# #                else 'neutral' for label in y_primary] # Clustering set
# lb = LabelBinarizer()
# if y_secondary:
#     lb.fit(y_secondary)
#     y = lb.transform(y_secondary)
# else:
#     lb.fit(y_primary)
#     y = lb.transform(y_primary)
# classes = lb.classes_
# num_classes = len(classes)
# y = y.reshape(-1, num_classes)
#
# # custom datasets
# dataset_size = len(X)
#
# # Separate hold-out fold from dataset
# X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
#
# X_training.reset_index(drop=True, inplace=True)
# X_test.reset_index(drop=True, inplace=True)
#
# cv_training_data = CNNTrainingDataset(
#     X_training,
#     y_training,
#     sample_freq=SAMPLING_FREQUENCY,
#     data_freq=DATASET_FREQUENCY,
#     lookback=LOOKBACK,
#     subset=SUBSET,
# )
# cv_test_data = CNNTrainingDataset(
#     X_test,
#     y_test,
#     sample_freq=SAMPLING_FREQUENCY,
#     data_freq=DATASET_FREQUENCY,
#     lookback=LOOKBACK,
#     subset=SUBSET,
# )
#
# test_dataloader = DataLoader(cv_test_data, batch_size=BATCH_SIZE, worker_init_fn=seed_worker, generator=g)
#
# # loss function
# criterion = nn.CrossEntropyLoss()
#
# # training loop
# history = {}
#
# for train_size in TRAIN_SIZES:
#     print(f"Train size: {train_size}")
#     split_history = {}
#
#     sss = StratifiedShuffleSplit(test_size=int(0.1 * dataset_size), train_size=int(train_size * dataset_size))
#     for i, (train_index, val_index) in enumerate(sss.split(X_training, y_training)):
#         # Initialize the model in each split
#         cnn_model = CNNSurfaceClassifier(input_size=INPUT_SIZE, output_size=num_classes).to(device)
#         # Initialize optimizer in each split
#         optimizer = torch.optim.Adam(
#             cnn_model.parameters(),
#             lr=1e-3,
#             eps=1e-6,
#             weight_decay=1e-3,
#         )
#         # Initialize scheduler in each split
#         scheduler = ExponentialLR(optimizer, gamma=0.9)
#         # Initialize early stopping
#         early_stopper = EarlyStopper()
#
#         train_dataloader = DataLoader(
#             Subset(cv_training_data, train_index),
#             batch_size=BATCH_SIZE,
#             worker_init_fn=seed_worker,
#             generator=g,
#             shuffle=True,
#         )
#         val_dataloader = DataLoader(
#             Subset(cv_training_data, val_index),
#             batch_size=BATCH_SIZE,
#             worker_init_fn=seed_worker,
#             generator=g,
#         )
#
#         train_batches = len(train_dataloader)
#         val_batches = len(val_dataloader)
#
#         for epoch in range(NUM_EPOCHS):
#             running_train_loss = 0.0
#             running_val_loss = 0.0
#
#             pbar = tqdm(train_dataloader, total=train_batches)
#             cnn_model.train()
#             for idx, (batch_x, batch_y) in enumerate(pbar):
#                 optimizer.zero_grad()
#
#                 batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#                 batch_x = batch_x.permute(0, 2, 1)
#                 train_outputs = cnn_model(batch_x)
#                 train_loss = criterion(train_outputs, batch_y)
#                 running_train_loss += train_loss
#
#                 # Backward pass
#                 train_loss.backward()
#                 optimizer.step()
#
#                 pbar.set_description(
#                     f"Fold {i + 1}/{sss.get_n_splits()},"
#                     f"Epoch {epoch + 1}/{NUM_EPOCHS},"
#                     f"Training loss: {running_train_loss / (idx + 1):.2E}"
#                 )
#             scheduler.step()
#
#             pbar_val = tqdm(val_dataloader, total=val_batches)
#             cnn_model.eval()
#             with torch.no_grad():
#                 for idx, (batch_x_val, batch_y_val) in enumerate(pbar_val):
#                     batch_x_val, batch_y_val = batch_x_val.to(device), batch_y_val.to(device)
#                     batch_x_val = batch_x_val.permute(0, 2, 1)
#                     val_outputs = cnn_model(batch_x_val)
#                     val_loss = criterion(val_outputs, batch_y_val)
#                     running_val_loss += val_loss
#
#                     pbar_val.set_description(
#                         f"Fold {i + 1}/{sss.get_n_splits()},"
#                         f"Epoch {epoch + 1}/{NUM_EPOCHS},"
#                         f"Validation loss: {running_val_loss / (idx + 1):.2E}"
#                     )
#
#             validation_loss = running_val_loss / (idx + 1)
#             if early_stopper.early_stop(validation_loss):
#                 print(f"Split {i + 1} ended on epoch {epoch + 1 - early_stopper.patience}!")
#                 break
#             if early_stopper.counter == 0:
#                 best_model = cnn_model.state_dict()
#
#         cnn_model.load_state_dict(best_model)
#
#         train_batches = len(train_dataloader)
#         y_true_train, y_pred_train = [], []
#         running_train_loss = 0.0
#
#         pbar_train = tqdm(train_dataloader, total=train_batches)
#         cnn_model.eval()
#         with torch.no_grad():
#             for idx, (batch_x_train, batch_y_train) in enumerate(pbar_train):
#                 batch_x_train, batch_y_train = batch_x_train.to(device), batch_y_train.to(device)
#                 batch_x_train = batch_x_train.permute(0, 2, 1)
#                 train_outputs = cnn_model(batch_x_train)
#                 train_loss = criterion(train_outputs, batch_y_train)
#                 running_train_loss += train_loss
#
#                 y_true_train.extend(torch.argmax(batch_y_train, dim=1).cpu().numpy())
#                 y_pred_train.extend(torch.argmax(train_outputs, dim=1).cpu().numpy())
#
#                 pbar_train.set_description(
#                     f"Fold {i + 1}/{sss.get_n_splits()}, Train loss: {running_train_loss / (idx + 1):.2E}")
#
#         test_batches = len(test_dataloader)
#         y_true_test, y_pred_test = [], []
#         running_test_loss = 0.0
#
#         pbar_test = tqdm(test_dataloader, total=test_batches)
#         cnn_model.eval()
#         with torch.no_grad():
#             for idx, (batch_x_test, batch_y_test) in enumerate(pbar_test):
#                 batch_x_test, batch_y_test = batch_x_test.to(device), batch_y_test.to(device)
#                 batch_x_test = batch_x_test.permute(0, 2, 1)
#                 test_outputs = cnn_model(batch_x_test)
#                 test_loss = criterion(test_outputs, batch_y_test)
#                 running_test_loss += test_loss
#
#                 y_true_test.extend(torch.argmax(batch_y_test, dim=1).cpu().numpy())
#                 y_pred_test.extend(torch.argmax(test_outputs, dim=1).cpu().numpy())
#
#                 pbar_test.set_description(
#                     f"Fold {i + 1}/{sss.get_n_splits()}, Test loss: {running_test_loss / (idx + 1):.2E}")
#
#         split_history[i + 1] = {
#             'train_accuracy': accuracy_score(y_true_train, y_pred_train),
#             'train_f1_score': f1_score(y_true_train, y_pred_train, average='macro'),
#             'test_accuracy': accuracy_score(y_true_test, y_pred_test),
#             'test_f1_score': f1_score(y_true_test, y_pred_test, average='macro'),
#         }
#     history[train_size] = split_history
#
# history_filename = '_'.join(CONFIGURATIONS + SUBSET) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S")
# json.dump(history, open(HISTORY_DIR / f'{history_filename}.json', 'w'))
#
# # set figure params
# nicer_green = '#159C48'
# nicer_blue = '#00A0FF'
# orange = '#FBBC04'
#
# plt.rcParams['figure.figsize'] = [5, 4]
# plt.rcParams["axes.prop_cycle"] = cycler('color', [nicer_blue, nicer_green, orange])
# plt.rcParams['lines.linewidth'] = 1.5
#
# # read data
# with open(HISTORY_DIR / f'{history_filename}.json') as fp:
#     history = json.load(fp)
#
with open('../results/biases/4W_imu_2024-06-21-04-02-42.json') as fp:
    history = json.load(fp)

nicer_green = '#159C48'
nicer_blue = '#00A0FF'
plt.rcParams['figure.figsize'] = [4, 3]
plt.rcParams["axes.prop_cycle"] = cycler('color', [nicer_blue, nicer_green])
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.size'] = 10

# plot results
train_curve, test_curve = [], []
for scores_dict in history.values():
    train_accuracy, test_accuracy = [], []
    for split in scores_dict.values():
        train_accuracy.append(split['train_accuracy'])
        test_accuracy.append(split['test_accuracy'])
    train_curve.append(np.mean(train_accuracy))
    test_curve.append(np.mean(test_accuracy))
plt.plot(TRAIN_SIZES, train_curve, label='train')
plt.plot(TRAIN_SIZES, test_curve, label='test')
plt.yticks(ticks=np.arange(4, 11) / 10)
plt.yticks(ticks=np.arange(45, 100, 5) / 100, minor=True)
plt.xlim(0.1, 0.8)
plt.ylim(0.4, 1)
plt.xlabel('train set size')
plt.ylabel('accuracy')
plt.grid(which='major', axis='both', linewidth=1)
plt.grid(which='minor', axis='y', linewidth=0.4)
plt.minorticks_on()
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'../results/biases/pessimistic_biases.png', dpi=300, bbox_inches="tight")
