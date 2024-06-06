import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import yaml

from scipy import signal, stats


def sample_sequence(dataframe, selected_columns, window_length, freq_stride, sampling_freq=None, differencing=False):
    """
    Draw random sample from run.
    Args:
        dataframe: data from run
        selected_columns: subset of columns representing chosen signal channels
        window_length: number of time steps in window
        freq_stride: how many time steps to omit from original data in window
        sampling_freq: if not None, parse this argument to highpass filter and filter data
        differencing: if True, calculate differences between consecutive time steps
    return: sequence dataframe containing selected channels and time steps
    """
    df = dataframe[selected_columns]
    init_timestep = random.randint(0, (len(dataframe) - 1) - window_length)
    sequence = df.iloc[init_timestep:init_timestep + window_length:freq_stride].to_numpy()
    if sampling_freq:
        sequence = butter_highpass_filter(sequence.T, 1, sampling_freq).T
    if differencing:
        sequence = np.diff(sequence, axis=0)[1:]
    return sequence


def sequence_run(dataframe, selected_columns, window_length, freq_stride, window_stride=1):
    """
    Convert dataframe to list of windows.
    Args:
        dataframe: data from run
        selected_columns: subset of columns representing chosen signal channels
        window_length: number of time steps in window
        freq_stride: how many time steps to omit from original data in window
        window_stride: how many time steps to omit from original data between windows
    return: list of dataframes containing selected channels and time steps
    """
    df = dataframe[selected_columns]
    num_of_windows = len(dataframe) - window_length + window_stride
    df_list = [df.iloc[init_timestep:init_timestep + window_length:freq_stride].to_numpy() for init_timestep in range(0, num_of_windows, window_stride)]
    return df_list


def butter_highpass_filter(data, cutoff, fs, order=5):
    """
    Args:
        data: signal to filter
        cutoff: filter cutoff frequency
        fs: signal sampling frequency
        order: filter order
    return: filtered signal
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False, output='ba')
    y = signal.filtfilt(b, a, data)
    return y


def plot_signal(dataframe, columns, title=None, y_label=None):
    """
    Args:
        dataframe: data from run
        columns: which signals to plot
        title: plot name
        y_label: y-axis name
    """

    plt.rcParams['figure.figsize'] = [15, 6]

    time = dataframe['Time'] - dataframe['Time'].min()
    for col in columns:
        plt.plot(
            time,
            dataframe[col],
            alpha=0.8,
            linewidth=0.8,
            marker='o',
            markersize=1,
            label=col,
        )
    if title:
        plt.title(title)
    plt.xlabel('time [$s$]')
    if y_label:
        plt.ylabel(y_label)
    plt.legend()
    plt.show()


def format_to_yaml(series):
    """
    Format string to yaml before unpacking.
    Args:
        series: data from run
    return: formatted data
    """
    data = series.tolist()
    data = [line.replace(', ', '];[') for line in data]
    data = [line.replace('[', '') for line in data]
    data = [line.replace(']', '') for line in data]
    data = [line.split(sep=';') for line in data]
    return data


def unpack_servo_data(dataframe):
    """
    Unpack data containing servo load to separate columns.
    Args:
        dataframe: data from run
    return: dataframe with servo load columns for each wheel
    """
    # convert strings to readable format
    data = format_to_yaml(dataframe['load'])
    # load yaml data
    data = [[yaml.safe_load(line) for line in separate_lines] for separate_lines in data]
    cols = ['Load_' + str(idx + 1) for idx in range(len(data[0]))]
    load = pd.DataFrame([[abs(dictionary['Amper']) / 100 for dictionary in line] for line in data], columns=cols)
    df = pd.concat([dataframe, load], axis=1)
    df.drop(columns=['load'], inplace=True)
    return df


def unpack_ang_vel_data(dataframe):
    """
    Unpack data containing angular velocities to separate columns.
    Args:
        dataframe: data from run
    return: dataframe with angular velocity columns for each wheel
    """
    # convert strings to readable format
    data = format_to_yaml(dataframe['angular_velocity'])
    # load yaml data
    data = [[yaml.safe_load(line) for line in separate_lines] for separate_lines in data]
    cols = ['AngVel_' + str(idx + 1) for idx in range(len(data[0]))]
    ang_vel = pd.DataFrame([[abs(dictionary['AngVelocity']) for dictionary in line] for line in data], columns=cols)
    df = pd.concat([dataframe, ang_vel], axis=1)
    df.drop(columns=['angular_velocity'], inplace=True)
    return df


def calculate_mean_power(load_dataframe, velocity_dataframe, old):
    """
    Calculate average power consumption for left and right side wheels based on index assignment.
    Args:
        load_dataframe: load data from run
        velocity_dataframe: angular velocity data from run
        old: index assignment type
    return: dataframe with average power consumption columns for each side
    """
    number_of_wheels = len(load_dataframe.columns)
    cols = ['Power_' + str(idx + 1) for idx in range(number_of_wheels)]
    df = pd.DataFrame(load_dataframe.values * velocity_dataframe.values, columns=cols, index=load_dataframe.index)
    if old:
        cols_left = ['Power_' + str(idx + 1) for idx in range(number_of_wheels) if idx in (0, 3)]
        cols_right = ['Power_' + str(idx + 1) for idx in range(number_of_wheels) if idx in (1, 2)]
    else:
        cols_left = ['Power_' + str(idx + 1) for idx in range(number_of_wheels) if idx % 2 == 0]
        cols_right = ['Power_' + str(idx + 1) for idx in range(number_of_wheels) if idx % 2 != 0]
    power_left = df[cols_left]
    power_right = df[cols_right]
    df['mean_power_left'] = power_left.mean(axis=1)
    df['mean_power_right'] = power_right.mean(axis=1)
    return df


def get_sample_features(sequence):
    "Get features for single sequence"

    time_features = get_time_domain(sequence)
    ft_features = get_frequency_domain(sequence)
    features = np.append(time_features, ft_features)
    return features


def get_time_domain(sequence):
    "How the signal changes in time"

    min_val = np.min(sequence, axis=0)
    max_val = np.max(sequence, axis=0)
    mean = np.mean(sequence, axis=0)
    std = np.std(sequence, axis=0)
    skewness = stats.skew(sequence, axis=0)
    kurtosis = stats.kurtosis(sequence, axis=0)

    # # other features
    rms = np.sqrt(np.mean(sequence ** 2, axis=0))
    peak = np.max(np.abs(sequence), axis=0)
    peak_to_peak = np.ptp(sequence, axis=0)  # the range between minimum and maximum values

    crest_factor = np.max(np.abs(sequence), axis=0) / np.sqrt(np.mean(sequence ** 2, axis=0))  # how extreme the peaks are in a waveform
    form_factor = np.sqrt(np.mean(sequence ** 2, axis=0)) / np.mean(sequence, axis=0)  # the ratio of the RMS (root mean square) value to the average value
    pulse_indicator = np.max(np.abs(sequence), axis=0) / np.mean(sequence, axis=0)

    features = np.array([min_val, max_val, mean, std, skewness, kurtosis, rms, peak, peak_to_peak, crest_factor, form_factor, pulse_indicator]).flatten()
    return features


def get_frequency_domain(sequence):
    "How much of the signal lies within each given frequency band over a range of frequencies"

    ft = np.fft.fft(sequence, axis=0)
    S = np.abs(ft ** 2) / len(sequence)

    ft_sum = np.sum(S, axis=0)
    ft_max = np.max(S, axis=0)
    ft_mean = np.mean(S, axis=0)
    ft_peak = np.max(np.abs(S), axis=0)
    ft_variance = np.var(S, axis=0)

    features = np.array([ft_sum, ft_max, ft_mean, ft_peak, ft_variance]).flatten()
    return features
