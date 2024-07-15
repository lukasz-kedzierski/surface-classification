import functools as ft
import json
import numpy as np
import os
import pandas as pd
import time

from pathlib import Path

# set variables
TRAIN = True
OLD = False
TOPIC_FILES = ('imu_acc_ar.csv', 'servo_1-motor_volAmp.csv', 'servo_2-motor_volAmp.csv', 'servo_3-motor_volAmp.csv', 'servo_4-motor_volAmp.csv')

main_dir = Path('../../data/testypodloz/bags/')
target_dir = Path('../../data/testypodloz/csv')
bag_paths = [folder for folder in main_dir.rglob('**/') if 'Przejazd' in folder.__str__()]

# loop over bags and process data
labels = {}
for bag_path in bag_paths:
    # split path into chunks
    subfolders = os.path.normpath(bag_path).split(os.sep)
    key = subfolders[-1] + '_' + str(time.time_ns())

    # # set servo indexing flag
    # if TRAIN:
    #     if subfolders[-5] == '4W':
    #         OLD = True

    # get cmd_vel, imu_data, odom, and servo_data
    bag_files = [bag_path.joinpath(bag_file) for bag_file in os.listdir(bag_path) if bag_file in TOPIC_FILES]

    # read and merge tables
    dataframes = [pd.read_csv(bag_file) for bag_file in bag_files]
    for i in range(1, 5):
        dataframes[i][f'Power{i}'] = dataframes[i]['Voltage'] * dataframes[i]['Amper']
        dataframes[i].drop(columns=['ID', 'Voltage', 'Amper'], inplace=True)
    dataframe = ft.reduce(lambda left, right: pd.merge(left, right, how='outer', on='Time'), dataframes)

    # clean resulting dataframe
    dataframe = dataframe[dataframe.columns.drop(list(dataframe.filter(regex='header')))]
    dataframe.columns = ['Time',
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
                         'Power1', 'Power2', 'Power3', 'Power4']
    dataframe = dataframe[['Time',
                           'linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z',
                           'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z',  # imu
                           'Power1', 'Power2', 'Power3', 'Power4']]  # servo

    new_timesteps = np.arange(dataframe['Time'].min(), dataframe['Time'].max(), 0.01)
    resampled_dataframe = pd.DataFrame()
    for col in dataframe.columns:
        df = dataframe[dataframe[col].notna()]
        resampled_dataframe[col] = np.interp(new_timesteps, df['Time'], df[col])

    resampled_dataframe['mean_power_left'] = resampled_dataframe[['Power1', 'Power3']].mean(axis=1)
    resampled_dataframe['mean_power_right'] = resampled_dataframe[['Power2', 'Power4']].mean(axis=1)
    resampled_dataframe.drop(columns=['Power1', 'Power2', 'Power3', 'Power4'], inplace=True)

    # set initial timestep at 0
    resampled_dataframe['Time'] -= resampled_dataframe['Time'].min()

    # remove gravity from acceleration wrt z axis
    resampled_dataframe['linear_acceleration.z'] -= resampled_dataframe['linear_acceleration.z'].mean()

    # write dataframe to csv
    resampled_dataframe.to_csv(target_dir.joinpath(key + '.csv'))

    # gather labels
    if TRAIN:
        sample_dict = {'surface': subfolders[-2]}
        labels[key] = sample_dict

# dump labels to json if gathered
if labels:
    with open('../../data/testypodloz/labels.json', 'w') as fp:
        json.dump(labels, fp)
