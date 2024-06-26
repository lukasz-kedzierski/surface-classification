import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cycler import cycler
from pathlib import Path

# set figure params
nicer_green = '#159C48'
nicer_blue = '#00A0FF'
orange = '#FBBC04'

plt.rcParams['figure.figsize'] = [5, 4]
plt.rcParams["axes.prop_cycle"] = cycler('color', [nicer_blue, nicer_green, orange])
plt.rcParams['lines.linewidth'] = 1.5

# set variables
HISTORY_DIR = Path('../results/cv/')
labels = ('IMU', 'POWER', 'IMU + POWER')
t_95 = 2.228

# select result files to plot
# filenames = (
#     'cv_3_4W_6W_imu_2024-05-27-20-47-56',
#     'cv_3_4W_6W_servo_2024-06-04-03-36-49',
#     'cv_3_4W_6W_imu_servo_2024-05-27-17-42-11',
# )
# figure_name = '3_4w_6w_300dpi'

# filenames = (
#     'cv_3_4W_imu_2024-05-27-22-17-37',
#     'cv_3_4W_servo_2024-06-04-05-11-19',
#     'cv_3_4W_imu_servo_2024-05-27-23-46-22',
# )
# figure_name = '3_4w_300dpi'

# filenames = (
#     'cv_3_6W_imu_2024-05-28-10-03-25',
#     'cv_3_6W_servo_2024-06-04-00-21-55',
#     'cv_3_6W_imu_servo_2024-05-28-01-33-03',
# )
# figure_name = '3_6w_300dpi'

# filenames = (
#     'cv_10_4W_6W_imu_2024-05-12-12-50-16',
#     'cv_10_4W_6W_servo_2024-06-04-12-19-43',
#     'cv_10_4W_6W_imu_servo_2024-05-12-06-19-23',
# )
# figure_name = '10_4w_6w_300dpi'

# filenames = (
#     'cv_10_4W_imu_2024-05-12-16-12-19',
#     'cv_10_4W_servo_2024-06-04-06-49-43',
#     'cv_10_4W_imu_servo_2024-05-12-14-35-37',
# )
# figure_name = '10_4w_300dpi'

filenames = (
    'cv_10_6W_imu_2024-05-12-20-11-19',
    'cv_10_6W_servo_2024-06-04-09-05-38',
    'cv_10_6W_imu_servo_2024-05-12-18-14-56',
)
figure_name = '10_6w_300dpi'

# read data from files
results = {}
for filename in filenames:
    with open(HISTORY_DIR / (filename + '.json')) as fp:
        results[filename] = json.load(fp)

# plot results
fig, ax = plt.subplots()

for result, label in zip(results.values(), labels):
    df = pd.DataFrame(result)
    res_array = np.array(df.loc['accuracy'].values.tolist()).T
    x = np.arange(1, res_array.shape[0] + 1)
    average_acc = res_array.mean(axis=1)
    std_dev_acc = res_array.std(axis=1)
    ci = t_95 * std_dev_acc / np.sqrt(10)
    ax.plot(x, average_acc)
    ax.fill_between(x, (average_acc - ci), (average_acc + ci), alpha=.2, label=label)
ax.set_xlim(1, 100)
ax.set_ylim(0.4, 1)
ax.set_xlabel('epoch')
ax.set_ylabel('average accuracy')
plt.grid(which='major', axis='both', linewidth=1)
plt.grid(which='minor', axis='both', linewidth=0.4)
plt.minorticks_on()
plt.legend(loc='lower right')
plt.savefig(f'../results/cv/{figure_name}.png', dpi=300, bbox_inches="tight")
plt.close()
