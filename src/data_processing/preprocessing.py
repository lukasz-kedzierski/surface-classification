"""Scripts to convert ROS bag files to CSV format and preprocess the data."""
import argparse
import functools as ft
import json
from pathlib import Path
import pandas as pd
from bagpy import bagreader
from tqdm import tqdm
from utils.processing import unpack_load_data, unpack_ang_vel_data, interpolate_servo_data, calculate_mean_power


def convert_rosbags(bag_dir):
    """Convert ROS bag files in the specified directory to CSV format.

    Parameters
    ----------
    bag_dir : pathlib.Path
        Path to the directory containing ROS bag files.
    """
    print("Extracting data from ROS bag files...")
    bag_paths = list(bag_dir.rglob('*.bag'))
    for bag_path in tqdm(bag_paths):
        b = bagreader(str(bag_path))
        for topic in b.topics:
            b.message_by_topic(topic)


def merge_csvs(bag_dir):
    """Merge CSV files from ROS bag data into a single dataframe per bag.

    Parameters
    ----------
    bag_dir : pathlib.Path
        Path to the directory containing ROS bag files.
    """
    print("\nProcessing data from ROS topics...")

    # Flag indicating whether servo indexing is circular (True) or alternating (False).
    # In alternating indexing left indexes are even and right indexes are odd.
    circular_indexing = False

    # Set working directories.
    target_dir = bag_dir.parents[0] / 'processed'
    target_dir.mkdir(parents=True, exist_ok=True)

    # Case specifiv method for gathering dataset directories.
    bag_paths = [folder for folder in bag_dir.rglob('*W/*') if folder.is_dir()]

    # Loop over bags and process data.
    labels = {}
    for bag_path in tqdm(bag_paths):
        # Split path into chunks.
        subfolders = bag_path.parts
        bag_name, kinematics, surface = subfolders[-1], subfolders[-2], subfolders[-3]

        # Set servo indexing flag.
        if kinematics == '4W':
            circular_indexing = True

        # Read IMU and servo data.
        imu_dataframe = pd.read_csv(bag_path.joinpath('imu-data.csv'))
        imu_time_reference = pd.read_csv(bag_path.joinpath('imu-time_ref.csv'))
        wheel_load_dataframe = pd.read_csv(bag_path.joinpath('Servo_data.csv'))
        wheel_velocity_dataframe = pd.read_csv(bag_path.joinpath('wheel_feedback.csv'))

        # Select IMU data rows that have timestamp reference.
        imu_dataframe = imu_dataframe[imu_dataframe['header.seq'].isin(imu_time_reference['header.seq'])]

        # Select servo data that lies within IMU data timeframe bounds.
        timeframe_min, timeframe_max = imu_dataframe['Time'].min(), imu_dataframe['Time'].max()
        wheel_load_dataframe = wheel_load_dataframe[(wheel_load_dataframe['Time'] >= timeframe_min) & (wheel_load_dataframe['Time'] <= timeframe_max)]
        wheel_velocity_dataframe = wheel_velocity_dataframe[(wheel_velocity_dataframe['Time'] >= timeframe_min) & (wheel_velocity_dataframe['Time'] <= timeframe_max)]

        # Unpack load and angular velocity values.
        wheel_load_dataframe = unpack_load_data(wheel_load_dataframe)
        wheel_velocity_dataframe = unpack_ang_vel_data(wheel_velocity_dataframe)

        # Adjust servo timesteps relative to IMU data.
        wheel_load_dataframe['Time'] -= timeframe_min
        wheel_velocity_dataframe['Time'] -= timeframe_min

        # Interpolate servo data to match IMU 100 Hz rate.
        new_timesteps = imu_time_reference['time_ref.secs'] + imu_time_reference['time_ref.nsecs'] / 1e9
        new_timesteps -= new_timesteps.min()
        imu_dataframe['Time'] = new_timesteps

        resampled_wheel_load_dataframe = interpolate_servo_data(wheel_load_dataframe, new_timesteps)
        resampled_wheel_velocity_dataframe = interpolate_servo_data(wheel_velocity_dataframe,
                                                                    new_timesteps)

        # Merge all data into a single dataframe.
        dataframes = [imu_dataframe,
                      resampled_wheel_load_dataframe,
                      resampled_wheel_velocity_dataframe]
        dataframe = ft.reduce(lambda left, right: pd.merge(left, right, how='outer', on='Time'),
                              dataframes)

        # Clean the resulting dataframe.
        imu_columns = ['linear_acceleration.x',
                       'linear_acceleration.y',
                       'linear_acceleration.z',
                       'angular_velocity.x',
                       'angular_velocity.y',
                       'angular_velocity.z']
        wheel_load_columns = [col for col in dataframe.columns if 'wheel_load' in col]
        wheel_angular_velocity_columns = [col for col in dataframe.columns if 'wheel_angular_velocity' in col]
        dataframe = dataframe[['Time'] + imu_columns + wheel_load_columns + wheel_angular_velocity_columns]

        # Trim the first and last 10% of rows for more coherent data.
        clip_var = int(len(dataframe) * .1)
        dataframe = dataframe.iloc[clip_var:-clip_var].reset_index(drop=True)
        dataframe['Time'] -= dataframe['Time'].min()

        # Center Z axis acceleration around 0.
        dataframe['linear_acceleration.z'] -= dataframe['linear_acceleration.z'].mean()

        # Calculate estimated power values.
        power = calculate_mean_power(
            dataframe[wheel_load_columns],
            dataframe[wheel_angular_velocity_columns],
            circular_indexing
            )
        dataframe = pd.concat([dataframe, power], axis=1)

        # Write dataframe to CSV file.
        dataframe.to_csv(target_dir.joinpath(bag_name + '.csv'))

        # If training set, gather labels.
        if 'train' in str(bag_dir):
            sample_dict = {'surface': surface, 'kinematics': kinematics}
            labels[bag_name] = sample_dict

    # Dump labels to JSON if gathered.
    if labels:
        with open(bag_dir.parents[0] / 'labels.json', 'w', encoding='utf-8') as fp:
            json.dump(labels, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert ROS bags to CSV and preprocess them.')
    parser.add_argument(
        '--bag_dir',
        type=Path,
        required=True,
        help='Path to dataset directory relative to root.'
    )
    args = parser.parse_args()

    convert_rosbags(args.bag_dir)
    merge_csvs(args.bag_dir)
