import pandas as pd

import torch
from torch.utils.data import Dataset

from utils import select_sequence


class SurfaceDataset(Dataset):
    def __init__(self, samples, labels, sample_freq=20., data_freq=100., lookback=1., subset=None):
        """
        Args:
            samples: array of time series, first dimension is number of time steps
            labels: array of surface labels
            sample_freq: lowest frequency of collected data
            data_freq: highest frequency of collected data
            lookback: size of window for prediction
            subset: list containing data categories
        """

        self.samples = samples
        self.labels = labels
        self.sampling_frequency = sample_freq
        self.dataset_frequency = data_freq
        self.lookback = int(lookback * self.sampling_frequency)
        self.stride = int(self.dataset_frequency / self.sampling_frequency)
        self.window_length = self.lookback * self.stride
        self.subset = subset
        if not self.subset:
            self.subset = ('cmd_vel', 'imu', 'odom', 'servo')
        measurements = {
            'cmd_vel': ['linear.x', 'angular.z'],
            'imu': ['linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z', 'angular_velocity.x',
                    'angular_velocity.y', 'angular_velocity.z'],
            'odom': ['pose.pose.position.x', 'pose.pose.position.y', 'twist.twist.linear.x', 'twist.twist.angular.z'],
            'servo': ['mean_power_left', 'mean_power_right'],
        }
        self.selected_columns = []
        for measurement in self.subset:
            self.selected_columns.extend(measurements[measurement])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve a single, windowed time series from dataset
        """

        print('Dataset idx: ', idx)
        sample = self.samples[idx]
        run = pd.read_csv(sample, index_col=[0]).drop(labels='Time', axis=1)

        X = select_sequence(run, self.selected_columns, self.window_length, self.stride)
        y = self.labels[idx]

        return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)
