"""Script to plot odometry error results from an .ods file."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.processing import generalize_classes
from utils.visualization import plot_odom_error, IMAGE_DIR


def plot_error(data: pd.DataFrame, image_path: Path) -> None:
    """Plot odometry error results from an .ods file.

    Parameters
    ----------
    data : pd.DataFrame
        Data read from the .ods file.
    image_path : pathlib.Path
        Path to save the output image.
    """

    # Process data.
    error_data = data['error']
    target_classes = data['surface']
    lb = LabelEncoder()
    generalized_classes = generalize_classes(target_classes)
    lb.fit(generalized_classes)
    assigned_labels = lb.transform(generalized_classes)

    # Plot odometry error.
    plot_odom_error(error_data, assigned_labels, image_path)


def main():
    """Main script for plotting odometry errors."""

    parser = argparse.ArgumentParser(description="Dataset Analysis for Surface Classification")
    parser.add_argument(
        '--output-dir',
        default='results',
        type=Path,
        help="Output directory path."
    )
    args = parser.parse_args()

    # Set up paths.
    image_dir = args.output_dir.joinpath(IMAGE_DIR)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Read data.
    data_path = Path('data/odom_error.ods')
    data = pd.read_excel(data_path,
                         sheet_name=None,
                         header=None,
                         names=['surface', 'error'],
                         engine="odf")

    for kinematics in ['4W', '6W']:
        image_path = image_dir.joinpath(f'odom_error_{kinematics}.png'.lower())
        plot_error(data[kinematics], image_path)


if __name__ == "__main__":
    main()
