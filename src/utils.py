import matplotlib.pyplot as plt
import pandas as pd
import random
import yaml

from scipy import signal


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
    df = df.iloc[init_timestep:init_timestep + window_length:freq_stride].to_numpy()
    if sampling_freq:
        df = butter_highpass_filter(df.T, 1, sampling_freq).T.copy()
    if differencing:
        df = df.diff().to_numpy()[1:]
    return df


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


def plot_signal(dataframe, columns, title):
    """
    Args:
        dataframe: data from run
        columns: which signals to plot
        title: plot name
    """
    plt.rcParams['figure.figsize'] = [15, 6]
    for col in columns:
        plt.plot(
            dataframe['Time'],
            dataframe[col],
            alpha=0.8,
            linewidth=0.8,
            marker='o',
            markersize=1,
            label=col,
        )
    plt.title(title)
    plt.xlabel('Time')
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
