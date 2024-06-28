import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import yaml

from cycler import cycler
from scipy import signal, stats

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

nicer_green = '#159C48'
nicer_blue = '#00A0FF'
orange = '#FBBC04'
pink = '#DB00CF'
mad_purple = '#732BF5'
light_green = '#66C2A5'
main_color = '#3282F6'  # zoom_plot background color

plt.rcParams['figure.figsize'] = [4, 3]


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
    df_list = [df.iloc[init_timestep:init_timestep + window_length:freq_stride].to_numpy()
               for init_timestep in range(0, num_of_windows, window_stride)]
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


def plot_signal(dataframe, columns, title=None, y_label=None, alpha=1):
    """
    Args:
        dataframe: data from run
        columns: which signals to plot
        title: plot name
        y_label: y-axis name
        alpha: plot transparency
    """

    plt.rcParams['figure.figsize'] = [8, 3]
    plt.rcParams["axes.prop_cycle"] = cycler('color', [nicer_blue, nicer_green, orange])
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['font.size'] = 10

    time = dataframe['Time'] - dataframe['Time'].min()
    for col in columns:
        plt.plot(
            time,
            dataframe[col],
            label=col,
            alpha=alpha,
        )
    if title:
        plt.title(title)
    plt.xlabel('time [s]')
    if y_label:
        plt.ylabel(y_label)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f'../../results/{y_label[:10]}.png', dpi=300, bbox_inches="tight")
    plt.show()


def plot_many(dataframe, columns, y_label=None, alpha=1):
    """
    Args:
        dataframe: data from run
        columns: which signals to plot
        y_label: y-axis name
        alpha: plot transparency
    """   
    plt.rcParams['figure.figsize'] = [8, 3]
    plt.rcParams["axes.prop_cycle"] = cycler('color', [nicer_blue, nicer_green, pink, orange])
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['font.size'] = 10

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

    time = dataframe['Time'] - dataframe['Time'].min()

    for i, ax in enumerate(axes.flat):
        col = columns[i]
        
        ax.plot(
            time,
            dataframe[col],
            label=col,
            alpha=alpha
        )
        ax.set_title(f"{col}")
            
    fig.supxlabel('time [s]')
    if y_label:
        fig.supylabel(y_label)
    plt.tight_layout()
    plt.savefig(f'../../results/{y_label[:10]}.png', dpi=300, bbox_inches="tight")
    plt.show()


def zoom_plot(dataframe, columns, y_label=None):
    """
    Args:
        dataframe: data from run
        columns: which signals to plot
        y_label: y-axis name
    """
    plt.rcParams['figure.figsize'] = [4, 3]
    plt.rcParams["axes.prop_cycle"] = cycler('color', [nicer_blue, nicer_green])
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['font.size'] = 10

    rejected_color = 'indianred'
    selected_color = 'darkgreen'

    # rectangle and connection patches coordinates
    origin_x, origin_y = 8, 0
    duration = 2.7
    height = 0.9
    alpha = 0.1

    rect = patches.Rectangle((origin_x, origin_y), duration, height, alpha=alpha, color=main_color)

    time = dataframe['Time'] - dataframe['Time'].min()
    zoom_start = time[time >= origin_x].index[0]  # 161
    zoom_end = time[time <= origin_x + 2.7].index[-1]  # 214

    # Create figures / axes
    fig = plt.figure()
    top_left = fig.add_subplot(2, 2, 1)
    top_left.set_xticks([])
    top_left.set_yticks([])
    top_left.patch.set_alpha(alpha)
    top_left.set_facecolor(main_color)

    # top_right = fig.add_subplot(2, 2, 2)
    # top_right.axis("off")

    bottom = fig.add_subplot(2, 1, 2)
    bottom.set_xlabel('time [s]')

    # fig.subplots_adjust(hspace=.55)

    for col in columns:
        bottom.plot(
            time,
            dataframe[col],
            label=col,
        )
    bottom.add_patch(rect)
    bottom.set_ylabel(y_label)
    # bottom.legend(['mean est. power left', 'mean est. power right'], loc="upper right")

    for col in columns:  # plot general figures
        top_left.plot(
            time.iloc[zoom_start:zoom_end],
            dataframe[col].iloc[zoom_start:zoom_end],
            label='_nolegend_',
        )

    col = columns[1]  # add markers
    marker_on = list(range(0, 53))
    top_left.plot(
        time.iloc[zoom_start:zoom_end],
        dataframe[col].iloc[zoom_start:zoom_end],
        label='selected samples',
        linewidth=0,  # turn off line visibility
        markevery=marker_on[::2],
        marker='o',
        markerfacecolor='darkgreen',
        markeredgecolor='darkgreen',
        markersize=2
    )

    top_left.plot(
        time.iloc[zoom_start:zoom_end],
        dataframe[col].iloc[zoom_start:zoom_end],
        label='discarded samples',
        linewidth=0,
        markevery=marker_on[1::2],
        marker='o',
        markerfacecolor='indianred',
        markeredgecolor='indianred',
        markersize=2
    )

    top_left.legend(loc='center left', bbox_to_anchor=[1.0, 0.5])

    # Add the connection patches
    fig.add_artist(patches.ConnectionPatch(
        xyA=(0, 0), coordsA=top_left.transAxes,  # small figure left point of tangency
        xyB=(origin_x, height), coordsB=bottom.transData,
        color='black'
    ))

    fig.add_artist(patches.ConnectionPatch(
        xyA=(1, 0), coordsA=top_left.transAxes,  # small figure left point of tangency
        xyB=(origin_x + duration, height), coordsB=bottom.transData,
        color='black'
    ))

    # plt.tight_layout()
    plt.savefig(r'../../results/zoom_plot.png', dpi=300, bbox_inches="tight")
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
    """Get features for single sequence"""

    time_features = get_time_domain(sequence)
    ft_features = get_frequency_domain(sequence)
    features = np.append(time_features, ft_features)
    return features


def get_time_domain(sequence):
    """How the signal changes in time"""

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

    crest_factor = peak / rms  # how extreme the peaks are in a waveform
    form_factor = rms / mean  # the ratio of the RMS (root mean square) value to the average value
    pulse_indicator = peak / mean

    features = np.array([
        min_val,
        max_val,
        mean,
        std,
        skewness,
        kurtosis,
        rms,
        peak,
        peak_to_peak,
        crest_factor,
        form_factor,
        pulse_indicator,
    ]).flatten()
    return features


def get_frequency_domain(sequence):
    """How much of the signal lies within each given frequency band over a range of frequencies"""

    ft = np.fft.fft(sequence, axis=0)
    S = np.abs(ft ** 2) / len(sequence)

    ft_sum = np.sum(S, axis=0)
    ft_max = np.max(S, axis=0)
    ft_mean = np.mean(S, axis=0)
    ft_peak = np.max(np.abs(S), axis=0)
    ft_variance = np.var(S, axis=0)

    features = np.array([ft_sum, ft_max, ft_mean, ft_peak, ft_variance]).flatten()
    return features
