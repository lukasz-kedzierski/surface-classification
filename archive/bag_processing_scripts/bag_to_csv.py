from bagpy import bagreader
from pathlib import Path

# set path to bag files
TRAIN = True
if TRAIN:
    MAIN_DIR = Path('../../data/train_set/fixed/')
else:
    MAIN_DIR = Path('../../data/test_set/bags/')
# MAIN_DIR = Path('../../data/testypodloz/bags/')

# read bags from directory
bag_paths = list(MAIN_DIR.rglob('*.bag'))
for bag_path in bag_paths:
    b = bagreader(bag_path.__str__())
    for topic in b.topics:
        b.message_by_topic(topic)
