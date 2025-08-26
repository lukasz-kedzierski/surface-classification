"""Script to convert ROS bag files to CSV format."""
import argparse
from pathlib import Path
from bagpy import bagreader
from tqdm import tqdm


def convert_rosbags(bag_dir):
    """Convert ROS bag files in the specified directory to CSV format.

    Parameters
    ----------
    bag_dir : pathlib.Path
        Path to the directory containing ROS bag files.
    """
    bag_paths = list(bag_dir.rglob('*.bag'))
    for bag_path in tqdm(bag_paths):
        b = bagreader(str(bag_path))
        for topic in b.topics:
            b.message_by_topic(topic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert ROS bag files to CSV.')
    parser.add_argument(
        '--bag_dir',
        type=Path,
        required=True,
        help='Path to dataset directory relative to root.'
        )
    args = parser.parse_args()

    convert_rosbags(args.bag_dir)
