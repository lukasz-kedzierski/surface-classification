import random

from scipy import signal


def select_sequence(dataframe, selected_columns, window_length, stride, sampling_frequency=None, differencing=False):
    """
    Draw random sample from run.
    Args:
        dataframe: data from run
        selected_columns: subset of columns representing chosen signal channels
        window_length: number of time steps in window
        stride: how many time steps to omit from original data
        sampling_frequency: if not None, parse this argument to highpass filter and filter data
        differencing: if True, calculate differences between consecutive time steps
    return: sequence dataframe containing selected channels and time steps
    """
    dataframe = dataframe[selected_columns]
    initial_timestep = random.randint(0, (len(dataframe) - 1) - window_length)
    final_timestep = initial_timestep + window_length
    timesteps = list(range(initial_timestep, final_timestep + stride, stride))
    dataframe = dataframe.iloc[timesteps].to_numpy()
    if sampling_frequency:
        dataframe = butter_highpass_filter(dataframe.T, 1, sampling_frequency).T.copy()
    if differencing:
        dataframe = dataframe.diff().to_numpy()[1:]
    return dataframe


def butter_highpass_filter(data, cutoff, fs, order=5):
    """
    Highpass filter.
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
