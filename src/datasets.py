import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from utils import get_sample_features, sample_sequence, sequence_run


class SurfaceDataset(Dataset):
    def __init__(
            self,
            runs: list | np.array,
            labels: np.array,
            data_freq: float = 100.,
            sample_freq: float | None = None,
            lookback: float = 1.,
            subset: list | None = None,
    ):
        """
        Dataset for training a CNN on several registered runs.
        Args:
            runs: list or array of time series file names
            labels: ndarray of one-hot encoded surface labels
            data_freq: frequency of collected data [Hz]
            sample_freq: new sampling frequency of windowed data [Hz]
            lookback: size of window for prediction [s]
            subset: list containing data sources
        """

        self.runs = runs
        self.labels = labels
        self.dataset_frequency = data_freq
        if sample_freq is None:
            self.sampling_frequency = data_freq
        else:
            self.sampling_frequency = sample_freq
        self.lookback = lookback * self.sampling_frequency

        self.stride = int(self.dataset_frequency / self.sampling_frequency)
        self.window_length = int(self.lookback * self.stride)

        if subset is None:
            self.subset = ('imu', 'servo')
        modalities = {
            'cmd_vel': ['linear.x', 'angular.z'],
            'imu': ['linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z',
                    'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z'],
            'odom': ['pose.pose.position.x', 'pose.pose.position.y', 'twist.twist.linear.x', 'twist.twist.angular.z'],
            'servo': ['mean_power_left', 'mean_power_right'],
        }
        self.selected_modalities = []
        for source in self.subset:
            self.selected_modalities.extend(modalities[source])

    def __len__(self):
        return len(self.runs)

    def __getitem__(self, idx):
        """
        Retrieves a single, windowed time series from dataset
        """

        run = self.runs[idx]
        time_series = pd.read_csv(run, index_col=[0]).drop(labels='Time', axis=1)

        X = sample_sequence(time_series, self.selected_modalities, self.window_length, self.stride)
        y = self.labels[idx]

        return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)


class SurfaceDatasetXGB(SurfaceDataset):
    def __init__(
            self,
            runs: list | np.array,
            labels: np.array,
            data_freq: float = 100.,
            sample_freq: float | None = None,
            lookback: float = 1.,
            subset: list | None = None,
    ):
        """
        Dataset for training XGBoost on several registered runs.
        Args:
            runs: list or array of time series file names
            labels: ndarray of one-hot encoded surface labels
            data_freq: frequency of collected data [Hz]
            sample_freq: new sampling frequency of windowed data [Hz]
            lookback: size of window for prediction [s]
            subset: list containing data sources
        """

        super().__init__(runs, labels, data_freq, sample_freq, lookback, subset)
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
        ) for channel in self.selected_modalities]
        engineered_freq_features = [channel + '_f_' + feature for feature in (
            'sum',
            'max',
            'mean',
            'peak',
            'var',
        ) for channel in self.selected_modalities]
        self.engineered_features = np.array(engineered_time_features + engineered_freq_features)

    def __getitem__(self, idx):
        """
        Retrieves a single, windowed time series from dataset
        """

        run = self.runs[idx]
        time_series = pd.read_csv(run, index_col=[0]).drop(labels='Time', axis=1)

        X = sample_sequence(time_series, self.selected_modalities, self.window_length, self.stride)
        X_hat = get_sample_features(X)
        y = self.labels[idx]

        return X_hat, y


class InferenceDataset(Dataset):
    def __init__(
            self,
            run: str,
            data_freq: float = 100.,
            sample_freq: float | None = None,
            lookback: float = 1.,
            subset: list | None = None,
    ):
        """
        Dataset for a CNN inference on a single registered run.
        Args:
            run: time series file name
            data_freq: frequency of collected data [Hz]
            sample_freq: new sampling frequency of windowed data [Hz]
            lookback: size of window for prediction [s]
            subset: list containing data sources
        """

        self.run = pd.read_csv(run, index_col=[0]).drop(labels='Time', axis=1)
        self.dataset_frequency = data_freq
        if sample_freq is None:
            self.sampling_frequency = data_freq
        else:
            self.sampling_frequency = sample_freq
        self.lookback = lookback * self.sampling_frequency

        self.stride = int(self.dataset_frequency / self.sampling_frequency)
        self.window_length = int(self.lookback * self.stride)

        if subset is None:
            self.subset = ('imu', 'servo')
        modalities = {
            'cmd_vel': ['linear.x', 'angular.z'],
            'imu': ['linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z', 'angular_velocity.x',
                    'angular_velocity.y', 'angular_velocity.z'],
            'odom': ['pose.pose.position.x', 'pose.pose.position.y', 'twist.twist.linear.x', 'twist.twist.angular.z'],
            'servo': ['mean_power_left', 'mean_power_right'],
        }
        self.selected_modalities = []
        for source in self.subset:
            self.selected_modalities.extend(modalities[source])

        self.samples = sequence_run(self.run, self.selected_modalities, self.window_length, self.stride)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves a single, windowed time series from dataset
        """

        X = self.samples[idx]

        return torch.tensor(X, dtype=torch.float)
