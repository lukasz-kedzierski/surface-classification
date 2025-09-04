import random
import numpy as np
import pandas as pd
import yaml
from scipy import stats


def sample_sequence(dataframe, selected_columns, window_length):
    """
    Draw random sample from run.
    Args:
        dataframe: data from run
        selected_columns: subset of columns representing chosen signal channels
        window_length: number of time steps in window
    return: sequence dataframe containing selected channels and time steps
    """
    df = dataframe[selected_columns]
    init_timestep = random.randint(0, (len(dataframe) - 1) - window_length)
    return df.iloc[init_timestep:init_timestep + window_length].to_numpy()


def sequence_run(dataframe, selected_columns, window_length, window_stride=1):
    """
    Convert dataframe to list of windows.
    Args:
        dataframe: data from run
        selected_columns: subset of columns representing chosen signal channels
        window_length: number of time steps in window
        window_stride: how many time steps to omit from original data between windows
    return: list of dataframes containing selected channels and time steps
    """
    df = dataframe[selected_columns]
    num_of_windows = len(dataframe) - window_length + window_stride
    df_list = [df.iloc[init_timestep:init_timestep + window_length].to_numpy()
               for init_timestep in range(0, num_of_windows, window_stride)]
    return df_list


def format_to_yaml(series):
    """Format string to yaml before unpacking.

    Parameters
    ----------
    series : pandas.Series
        Series containing string data to be formatted.

    Returns
    -------
    data : list of lists
        List of lists where each inner list contains formatted data from the series.
    """
    data = series.tolist()
    data = [line.replace(', ', '];[') for line in data]
    data = [line.replace('[', '') for line in data]
    data = [line.replace(']', '') for line in data]
    data = [line.split(sep=';') for line in data]
    return data


def unpack_load_data(dataframe):
    """Unpack data containing servo load to separate columns.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing column with servo load data.

    Returns
    -------
    pandas.DataFrame
        Dataframe with servo load columns for each wheel.
    """
    # Convert strings to readable format.
    data = format_to_yaml(dataframe['values'])
    # Load YAML data.
    data = [[yaml.safe_load(line) for line in separate_lines] for separate_lines in data]
    cols = ['wheel_load.' + str(idx + 1) for idx in range(len(data[0]))]
    # Load is divided by 100 because it is a percentage value.
    load = pd.DataFrame(
        [[abs(dictionary['Amper']) / 100 for dictionary in line] for line in data],
        columns=cols
        )
    df = pd.concat([dataframe, load], axis=1)
    df.drop(columns=['values'], inplace=True)
    return df


def unpack_ang_vel_data(dataframe):
    """Unpack data containing wheel angular velocities to separate columns.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing column with wheel angular velocity data.

    Returns
    -------
    pandas.DataFrame
        Dataframe with angular velocity columns for each wheel.
    """
    # Convert strings to readable format.
    data = format_to_yaml(dataframe['values'])
    # Load YAML data.
    data = [[yaml.safe_load(line) for line in separate_lines] for separate_lines in data]
    cols = ['wheel_angular_velocity.' + str(idx + 1) for idx in range(len(data[0]))]
    ang_vel = pd.DataFrame(
        [[abs(dictionary['AngVelocity']) for dictionary in line] for line in data],
        columns=cols
        )
    df = pd.concat([dataframe, ang_vel], axis=1)
    df.drop(columns=['values'], inplace=True)
    return df


def interpolate_servo_data(dataframe, new_timesteps):
    """Interpolate data.

    Parameters
    ----------
    dataframe : pandas DataFrame
        Dataframe containing original samples. Requires 'Time' column as the first one.
    new_timesteps : pandas.Series
        New timestamps for interpolation.

    Returns
    -------
    resampled_dataframe : pandas.DataFrame
        DataFrame containing samples mapped to new timestamps.
    """
    resampled_dataframe = pd.DataFrame(new_timesteps, columns=['Time'])
    for col in dataframe.columns[1:]:
        resampled_dataframe[col] = np.interp(new_timesteps, dataframe['Time'], dataframe[col])

    return resampled_dataframe


def calculate_mean_power(wheel_load_dataframe, wheel_velocity_dataframe, circular_indexing):
    """Calculate average power consumption for left and right side wheels based on index assignment.

    Parameters
    ----------
    wheel_load_dataframe : pandas.DataFrame
        Wheel load data from run.
    wheel_velocity_dataframe : pandas.DataFrame
        Wheel angular velocity data from run.
    circular_indexing : bool
        Wheel indexing format.

    Returns
    pandas.DataFrame
        Dataframe with average power consumption columns for each side.
    """
    number_of_wheels = len(wheel_load_dataframe.columns)
    cols = ['estimated_power.' + str(idx + 1) for idx in range(number_of_wheels)]
    df = pd.DataFrame(
        wheel_load_dataframe.values * wheel_velocity_dataframe.values,
        columns=cols,
        index=wheel_load_dataframe.index
        )
    if circular_indexing:
        cols_left = ['estimated_power.' + str(idx + 1) for idx in range(number_of_wheels) if idx in (0, 3)]
        cols_right = ['estimated_power.' + str(idx + 1) for idx in range(number_of_wheels) if idx in (1, 2)]
    else:
        cols_left = ['estimated_power.' + str(idx + 1) for idx in range(number_of_wheels) if idx % 2 == 0]
        cols_right = ['estimated_power.' + str(idx + 1) for idx in range(number_of_wheels) if idx % 2 != 0]
    power_left = df[cols_left]
    power_right = df[cols_right]
    df['mean_power_left'] = power_left.mean(axis=1)
    df['mean_power_right'] = power_right.mean(axis=1)
    return df


def get_sample_features(sequence, time_features=None, freq_features=None):
    """Get features for single sequence.

    Parameters
    ----------
    sequence : ndarray
        Time series data for which features are to be extracted.
    time_features : list of str, optional
        List of time domain features to extract. If None, no time features are extracted.
    freq_features : list of str, optional
        List of frequency domain features to extract. If None, no frequency features are extracted.

    Returns
    -------
    ndarray
        Extracted features as a flattened array.
    """
    if time_features is None and freq_features is None:
        raise ValueError("At least one of time_features or freq_features must be provided.")

    engineered_features = []
    if time_features is not None:
        engineered_features.extend(get_time_domain(sequence, time_features))
    if freq_features is not None:
        engineered_features.extend(get_frequency_domain(sequence, freq_features))

    return np.array(engineered_features)


def get_time_domain(sequence, time_features):
    """Get time features.

    Parameters
    ----------
    sequence : ndarray
        Time series data for which time features are to be extracted.
    time_features : list of str
        List of time domain features to extract.

    Returns
    -------
    ndarray
        Extracted time features as a flattened array.
    """
    engineered_time_features = []

    if 'min' in time_features:
        engineered_time_features.append(np.min(sequence, axis=0))
    if 'max' in time_features:
        engineered_time_features.append(np.max(sequence, axis=0))
    if 'mean' in time_features:
        mean = np.mean(sequence, axis=0)
        engineered_time_features.append(mean)
    if 'std' in time_features:
        engineered_time_features.append(np.std(sequence, axis=0))
    if 'skew' in time_features:
        engineered_time_features.append(stats.skew(sequence, axis=0))
    if 'kurt' in time_features:
        engineered_time_features.append(stats.kurtosis(sequence, axis=0))

    if 'rms' in time_features:
        rms = np.sqrt(np.mean(sequence ** 2, axis=0))
        engineered_time_features.append(rms)
    if 'peak' in time_features:
        peak = np.max(np.abs(sequence), axis=0)
        engineered_time_features.append(peak)
    if 'p2p' in time_features:
        engineered_time_features.append(np.ptp(sequence, axis=0))

    if 'crest' in time_features:
        engineered_time_features.append(peak / rms)  # how extreme the peaks are in a waveform
    if 'form' in time_features:
        engineered_time_features.append(rms / mean)  # the ratio of the RMS (root mean square) value to the average value
    if 'pulse' in time_features:
        engineered_time_features.append(peak / mean)

    return np.array(engineered_time_features).flatten()


def get_frequency_domain(sequence, freq_features):
    """Get frequency features.

    Parameters
    ----------
    sequence : ndarray
        Time series data for which frequency features are to be extracted.
    freq_features : list of str
        List of frequency domain features to extract.

    Returns
    -------
    ndarray
        Extracted frequency features as a flattened array.
    """
    engineered_freq_features = []

    ft = np.fft.fft(sequence, axis=0)
    s = np.abs(ft ** 2) / len(sequence)

    if 'sum' in freq_features:
        engineered_freq_features.append(np.sum(s, axis=0))
    if 'max' in freq_features:
        engineered_freq_features.append(np.max(s, axis=0))
    if 'mean' in freq_features:
        engineered_freq_features.append(np.mean(s, axis=0))
    if 'peak' in freq_features:
        engineered_freq_features.append(np.max(np.abs(s), axis=0))
    if 'var' in freq_features:
        engineered_freq_features.append(np.var(s, axis=0))

    return np.array(engineered_freq_features).flatten()


def generalize_classes(surface_classes):
    """Generalize surface classes.

    Parameters
    ----------
    surface_classes : list of str
        List of original surface classes.

    Returns
    -------
    list of str
        List of generalized surface classes.
    """
    return ['slippery' if label in ('3_Wykladzina_jasna', '4_Trawa')
            else 'grippy' if label in ('5_Spienione_PCV', '8_Pusta_plyta',
                                       '9_podklady', '10_Mata_ukladana')
            else 'neutral' for label in surface_classes]
