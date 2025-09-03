import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from utils.processing import unpack_load_data, unpack_ang_vel_data, calculate_mean_power
from utils.training import load_config
from utils.visualization import plot_signal, plot_many


def analyze_run(bag_path):
    """Local dataset analysis comprising of one run."""
    # Read IMU and servo data.
    imu_dataframe = pd.read_csv(bag_path.joinpath('imu-data.csv'))
    wheel_load_dataframe = pd.read_csv(bag_path.joinpath('Servo_data.csv'))
    wheel_velocity_dataframe = pd.read_csv(bag_path.joinpath('wheel_feedback.csv'))

    imu_lin_acc_cols = ['linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z']
    imu_ang_vel_cols = ['angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z']

    # Unpack load and angular velocity values.
    wheel_load_dataframe = unpack_load_data(wheel_load_dataframe)
    wheel_velocity_dataframe = unpack_ang_vel_data(wheel_velocity_dataframe)

    wheel_angular_velocity_cols = [col for col in wheel_velocity_dataframe.columns if 'wheel_angular_velocity' in col]
    wheel_load_cols = [col for col in wheel_load_dataframe.columns if 'wheel_load' in col]

    # Calculate estimated power values.
    power_dataframe = calculate_mean_power(
        wheel_load_dataframe[wheel_load_cols],
        wheel_velocity_dataframe[wheel_angular_velocity_cols],
        True
        )

    power_cols = [col for col in power_dataframe.columns if 'estimated_power' in col]

    # Plot all manuscript figures.
    plot_signal(
        imu_dataframe,
        imu_lin_acc_cols,
        y_label='IMU acceleration [$\mathregular{m/s^2}$]',
        alpha=0.5
        )
    plot_signal(
        imu_dataframe,
        imu_ang_vel_cols,
        y_label='IMU angular velocity [$\mathregular{rad/s}$]',
        alpha=0.5
        )
    plot_many(
        wheel_velocity_dataframe,
        wheel_angular_velocity_cols,
        y_label='servo angular velocity [rpm/min]'
        )
    plot_many(
        wheel_load_dataframe,
        wheel_load_cols,
        y_label='servo fractional load'
        )
    plot_signal(power_dataframe, power_cols, y_label='power estimate')


def analyze_dataset(dataset_params, output_dir):
    """Global dataset analysis including:
        1. mutual information score between features and target,
        2. feature correlation analysis.
    """
    data_dir = Path(dataset_params['data_dir'])
    labels_file = Path(dataset_params['labels_file'])

    image_dir = output_dir.joinpath('images')

    with open(labels_file, encoding='utf-8') as fp:
        labels = json.load(fp)

    dataset = [(data_dir.joinpath(key + '.csv'), values['surface'])
               for key, values in labels.items()]

    filenames = [run[0] for run in dataset]
    target_classes = [run[1] for run in dataset]

    le = LabelEncoder()
    if dataset_params['generalized_classes']:
        generalized_classes = [
            'slippery' if label in ('3_Wykladzina_jasna', '4_Trawa')
            else 'grippy' if label in ('5_Spienione_PCV', '8_Pusta_plyta', '9_podklady', '10_Mata_ukladana')
            else 'neutral' for label in target_classes]
        le.fit(generalized_classes)
        y = le.transform(generalized_classes)
    else:
        le.fit(target_classes)
        y = le.transform(target_classes)

    df_list = [pd.read_csv(file, index_col=[0]) for file in filenames]

    x = pd.concat(df_list, axis=0)

    mutual_information_score = mutual_info_classif(x.drop(columns=['Time']), y)
    mutual_information_df = pd.DataFrame(
        mutual_information_score.reshape(1, -1),
        columns=x.columns[1:],
        index=['Mutual Information']
        )
    mutual_information_df.to_json(output_dir / 'mutual_information.json')

    plt.rcParams['figure.figsize'] = [4, 3]
    plt.rcParams['font.size'] = 8
    corr_matrix = x.drop(columns=['Time']).corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, vmin=-1, vmax=1, cmap="Blues", annot=True, linewidths=0.5, fmt=".2f")
    plt.xticks(rotation=45, ha='right')
    plt.savefig(image_dir / 'corr_plot.png', dpi=300, bbox_inches="tight")


def main():
    """Main script for dataset analysis."""
    parser = argparse.ArgumentParser(description="Dataset Analysis for Surface Classification")
    parser.add_argument(
        '--config',
        type=Path,
        help="YAML configuration file path"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help="Output directory path"
    )
    args = parser.parse_args()

    analysis_params = load_config(args.config)
    dataset_params = analysis_params['dataset_params']
    bag_path = analysis_params['bag_path']

    analyze_run(bag_path)
    analyze_dataset(dataset_params, args.output_dir)


if __name__ == "__main__":
    main()