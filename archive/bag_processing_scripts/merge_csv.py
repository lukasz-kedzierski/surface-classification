import functools as ft
import json
import os
import pandas as pd

from pathlib import Path

from src.utils import unpack_servo_data, unpack_ang_vel_data, calculate_mean_power

# set variables
TRAIN = False
OLD = False
TOPIC_FILES = ('imu-data.csv', 'Servo_data.csv', 'wheel_feedback.csv')
# set containing velocity commands and odometry:
# ('cmd_vel.csv', 'imu-data.csv', 'odom.csv', 'Servo_data.csv', 'wheel_feedback.csv')

# get all bag paths
if TRAIN:
    main_dir = Path('../../data/train_set/fixed/')
    target_dir = Path('../../data/train_set/csv')
    bag_paths = [folder for folder in main_dir.rglob('**/') if
                 any(subfolder in folder.__str__() for subfolder in ('T1\\', 'T2\\'))]
else:
    main_dir = Path('../../data/test_set/bags/')
    target_dir = Path('../../data/test_set/csv/')
    bag_paths = [folder for folder in main_dir.rglob('**/')][1:]

# loop over bags and process data
labels = {}
for bag_path in bag_paths:
    # split path into chunks
    subfolders = os.path.normpath(bag_path).split(os.sep)
    key = subfolders[-1]

    # set servo indexing flag
    if TRAIN:
        if subfolders[-5] == '4W':
            OLD = True

    # get cmd_vel, imu_data, odom, and servo_data
    bag_files = [bag_path.joinpath(bag_file) for bag_file in os.listdir(bag_path) if bag_file in TOPIC_FILES]

    # read and merge tables
    dataframes = [pd.read_csv(bag_file) for bag_file in bag_files]
    dataframes[-2].rename(columns={"values": "load"})
    dataframes[-1].rename(columns={"values": "angular_velocity"})
    dataframe = ft.reduce(lambda left, right: pd.merge(left, right, how='outer', on='Time'), dataframes)

    # clean resulting dataframe
    dataframe = dataframe[dataframe.columns.drop(list(dataframe.filter(regex='header')))]
    dataframe.columns = ['Time',
                         # 'linear.x', 'linear.y', 'linear.z',  # linear velocity commands
                         # 'angular.x', 'angular.y', 'angular.z',    # angular velocity commands
                         'orientation.x', 'orientation.y', 'orientation.z', 'orientation.w',
                         'orientation_covariance_0', 'orientation_covariance_1', 'orientation_covariance_2',
                         'orientation_covariance_3', 'orientation_covariance_4', 'orientation_covariance_5',
                         'orientation_covariance_6', 'orientation_covariance_7', 'orientation_covariance_8',
                         'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z',
                         'angular_velocity_covariance_0', 'angular_velocity_covariance_1',
                         'angular_velocity_covariance_2',
                         'angular_velocity_covariance_3', 'angular_velocity_covariance_4',
                         'angular_velocity_covariance_5',
                         'angular_velocity_covariance_6', 'angular_velocity_covariance_7',
                         'angular_velocity_covariance_8',
                         'linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z',
                         'linear_acceleration_covariance_0', 'linear_acceleration_covariance_1',
                         'linear_acceleration_covariance_2',
                         'linear_acceleration_covariance_3', 'linear_acceleration_covariance_4',
                         'linear_acceleration_covariance_5',
                         'linear_acceleration_covariance_6', 'linear_acceleration_covariance_7',
                         'linear_acceleration_covariance_8',
                         # 'child_frame_id', 'pose.pose.position.x', 'pose.pose.position.y', 'pose.pose.position.z',
                         # 'pose.pose.orientation.x', 'pose.pose.orientation.y', 'pose.pose.orientation.z',
                         # 'pose.pose.orientation.w', 'pose.covariance',
                         # 'twist.twist.linear.x', 'twist.twist.linear.y', 'twist.twist.linear.z',
                         # 'twist.twist.angular.x', 'twist.twist.angular.y', 'twist.twist.angular.z',
                         # 'twist.covariance',    # odometry
                         'load', 'angular_velocity']
    dataframe = dataframe[['Time',
                           # 'linear.x', 'angular.z', # cmd_vel
                           'linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z',
                           'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z',  # imu
                           # 'pose.pose.position.x', 'pose.pose.position.y',
                           # 'twist.twist.linear.x', 'twist.twist.angular.z', # odom
                           'load', 'angular_velocity']]  # servo

    # fill missing values
    dataframe.ffill(inplace=True)

    # trim first and last rows for more coherent data
    clip_var = int(len(dataframe) * .1)
    dataframe = dataframe.iloc[clip_var:-clip_var].reset_index(drop=True)

    # set initial timestep at 0
    dataframe['Time'] -= dataframe['Time'].min()

    # remove gravity from acceleration wrt z axis
    dataframe['linear_acceleration.z'] -= dataframe['linear_acceleration.z'].mean()

    # unpack load values
    dataframe = unpack_servo_data(dataframe)

    # unpack angular velocity values
    dataframe = unpack_ang_vel_data(dataframe)

    # calculate power values
    load_cols = [col for col in dataframe.columns if 'Load' in col]
    velocity_cols = [col for col in dataframe.columns if 'Vel' in col]
    power = calculate_mean_power(dataframe[load_cols], dataframe[velocity_cols], OLD)
    dataframe = pd.concat([dataframe, power], axis=1)

    # write dataframe to csv
    dataframe.to_csv(target_dir.joinpath(key + '.csv'))

    # gather labels
    if TRAIN:
        sample_dict = {'surface': subfolders[-6], 'kinematics': subfolders[-5], 'spacing': subfolders[-4],
                       'angle': subfolders[-3], 'trajectory': subfolders[-2]}
        labels[key] = sample_dict

# dump labels to json if gathered
if labels:
    with open('../data/train_set/labels.json', 'w') as fp:
        json.dump(labels, fp)
