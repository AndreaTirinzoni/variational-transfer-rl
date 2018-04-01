import utils
import numpy as np
import matplotlib.pyplot as plt

MARKERS = ["o", "D", "s", "^", "v", "p", "*"]
COLORS = ["#0e5ad3", "#bc2d14", "#22aa16", "#a011a3", "#d1ba0e", "#14ccc2", "#d67413"]
LINES = ["solid", "dashed", "dashdot", "dotted", "solid", "dashed", "dashdot", "dotted"]


def plot_curves(x_data, y_mean_data, y_std_data=None, title="", x_label="Episodes", y_label="Performance", names=None,
                file_name=None):
    assert len(x_data) < 8

    plt.style.use('ggplot')

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 20
    # plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 22
    plt.rcParams['figure.titlesize'] = 20

    fig, ax = plt.subplots()

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    X = np.array(x_data)
    plt.xlim([X.min(), X.max()])

    for i in range(len(x_data)):

        ax.plot(x_data[i], y_mean_data[i], linewidth=4, color=COLORS[i], marker=MARKERS[i], markersize=8.0,
                linestyle=LINES[i], label=names[i] if names is not None else None)
        if y_std_data is not None:
            ax.fill_between(x_data[i], y_mean_data[i] - y_std_data[i], y_mean_data[i] + y_std_data[i],
                            facecolor=COLORS[i], edgecolor=COLORS[i], alpha=0.3)

    if names is not None:
        ax.legend(loc='best', numpoints=1, fancybox=True, frameon=False)

    if file_name is not None:
        plt.savefig(file_name + ".pdf", format='pdf')

    plt.show()


files = ["mm"]

x = []
y_mean = []
y_std = []

for file in files:
    results = utils.load_object(file)
    iterations = []
    n_samples = []
    rewards = []
    l_2 = []
    l_inf = []
    for result in results:
        iterations.append(result[0])
        n_samples.append(result[1])
        rewards.append([r[0] for r in result[2]])
        l_2.append(result[3])
        l_inf.append(result[4])
    iterations = np.array(iterations)
    x.append(np.mean(iterations,axis=0))
    n_samples = np.array(n_samples)
    rewards = np.array(rewards)
    y_mean.append(np.mean(rewards, axis=0))
    y_std.append(np.std(rewards, axis=0) / np.sqrt(rewards.shape[0]))
    l_2 = np.array(l_2)
    l_inf = np.array(l_inf)

plot_curves(x, y_mean, y_std, title="", x_label="Iterations", y_label="Reward", names=None, file_name=None)
