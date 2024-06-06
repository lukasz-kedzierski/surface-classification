import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from utils import get_sample_features, sample_sequence, sequence_run


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
        self.lookback = lookback * self.sampling_frequency
        self.stride = int(self.dataset_frequency / self.sampling_frequency)
        self.window_length = int(self.lookback * self.stride)
        self.subset = subset
        if not self.subset:
            self.subset = ('imu', 'servo')
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
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a single, windowed time series from dataset
        """

        sample = self.samples[idx]
        run = pd.read_csv(sample, index_col=[0]).drop(labels='Time', axis=1)

        X = sample_sequence(run, self.selected_columns, self.window_length, self.stride)
        y = self.labels[idx]

        return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)


class SurfaceDatasetXGB(SurfaceDataset):
    def __init__(self, samples, labels, sample_freq=20., data_freq=100., lookback=1., subset=None):
        super().__init__(samples, labels, sample_freq, data_freq, lookback, subset)
        engineered_time_features = [channel + '_t_' + feature for feature in (
            'min',
            'max',
            'mean',
            'std',
            'skew',
            'kurt',
            'rms',
            'peak',
            'p2p',
            'crest',
            'form',
            'pulse',
        ) for channel in self.selected_columns]
        engineered_freq_features = [channel + '_f_' + feature for feature in (
            'sum',
            'max',
            'mean',
            'peak',
            'var',
        ) for channel in self.selected_columns]
        self.engineered_features = np.array(engineered_time_features + engineered_freq_features)

    def __getitem__(self, idx):
        """
        Retrieve a single, windowed time series from dataset
        """

        sample = self.samples[idx]
        run = pd.read_csv(sample, index_col=[0]).drop(labels='Time', axis=1)

        X = sample_sequence(run, self.selected_columns, self.window_length, self.stride)
        X_hat = get_sample_features(X)
        y = self.labels[idx]

        return X_hat, y


class InferenceDataset(Dataset):
    def __init__(self, run, sample_freq=20., data_freq=100., lookback=1., subset=None):
        """
        Args:
            run: array of time series, first dimension is number of time steps
            sample_freq: lowest frequency of collected data
            data_freq: highest frequency of collected data
            lookback: size of window for prediction
            subset: list containing data categories
        """

        self.run = pd.read_csv(run, index_col=[0]).drop(labels='Time', axis=1)
        self.sampling_frequency = sample_freq
        self.dataset_frequency = data_freq
        self.lookback = lookback * self.sampling_frequency
        self.stride = int(self.dataset_frequency / self.sampling_frequency)
        self.window_length = int(self.lookback * self.stride)
        self.subset = subset
        if not self.subset:
            self.subset = ('imu', 'servo')
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

        self.samples = sequence_run(self.run, self.selected_columns, self.window_length, self.stride)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a single, windowed time series from dataset
        """

        X = self.samples[idx]

        return torch.tensor(X, dtype=torch.float)
