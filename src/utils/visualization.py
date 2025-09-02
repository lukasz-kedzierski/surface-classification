import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import patches

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams['figure.figsize'] = [4, 3]

nicer_green = '#159C48'
nicer_blue = '#00A0FF'
orange = '#FBBC04'
pink = '#DB00CF'
mad_purple = '#732BF5'
light_green = '#66C2A5'
main_color = '#3282F6'  # zoom_plot background color


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
