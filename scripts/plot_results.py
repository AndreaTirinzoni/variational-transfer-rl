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

        ax.plot(x_data[i], y_mean_data[i], linewidth=3, color=COLORS[i], marker=None, markersize=8.0,
                linestyle="solid", label=names[i] if names is not None else None)
        if y_std_data is not None:
            ax.fill_between(x_data[i], y_mean_data[i] - y_std_data[i], y_mean_data[i] + y_std_data[i],
                            facecolor=COLORS[i], edgecolor=COLORS[i], alpha=0.3)

    if names is not None:
        ax.legend(loc='best', numpoints=1, fancybox=True, frameon=False)

    if file_name is not None:
        plt.savefig(file_name + ".pdf", format='pdf')

    plt.show()


files = ["nt_gw5x5","vt_gw5x5_lambda0.001_tc0","vt_gw5x5_lambda0.0001_tc0","vt_gw5x5_lambda0.001_tc1"]
names = ["No Transfer", "Transfer (0.001)", "Transfer (0.0001)", "Transfer (TC)"]

x = []
y_mean = []
y_std = []
y2_mean = []
y2_std = []
y3_mean = []
y3_std = []
y4_mean = []
y4_std = []
y5_mean = []
y5_std = []

for file in files:
    results = [r[2] for r in utils.load_object(file)]
    iterations = []
    episodes = []
    n_samples = []
    lear_rew = []
    eval_rew = []
    l_2 = []
    l_inf = []
    sft = []
    for result in results:
        iterations.append(result[0])
        episodes.append(result[1])
        n_samples.append(result[2])
        lear_rew.append(result[3])
        eval_rew.append(result[4])
        l_2.append(result[5])
        l_inf.append(result[6])
        sft.append(result[7])
    iterations = np.array(iterations)
    x.append(np.mean(iterations,axis=0))
    n_samples = np.array(n_samples)
    lear_rew = np.array(lear_rew)
    y_mean.append(np.mean(lear_rew, axis=0))
    y_std.append(2 * np.std(lear_rew, axis=0) / np.sqrt(lear_rew.shape[0]))
    eval_rew = np.array(eval_rew)
    y2_mean.append(np.mean(eval_rew, axis=0))
    y2_std.append(2 * np.std(eval_rew, axis=0) / np.sqrt(eval_rew.shape[0]))
    l_2 = np.array(l_2)
    y3_mean.append(np.mean(l_2, axis=0))
    y3_std.append(2 * np.std(l_2, axis=0) / np.sqrt(l_2.shape[0]))
    l_inf = np.array(l_inf)
    y4_mean.append(np.mean(l_inf, axis=0))
    y4_std.append(2 * np.std(l_inf, axis=0) / np.sqrt(l_inf.shape[0]))
    sft = np.array(sft)
    y5_mean.append(np.mean(sft, axis=0))
    y5_std.append(2 * np.std(sft, axis=0) / np.sqrt(sft.shape[0]))

plot_curves([a[1:] for a in x], [a[1:] for a in y_mean], [a[1:] for a in y_std], title="", x_label="Iterations", y_label="Learning Reward", names=names, file_name="lrev")
plot_curves(x, y2_mean, y2_std, title="", x_label="Iterations", y_label="Evaluation Reward", names=names, file_name="erew")
plot_curves(x, y3_mean, y3_std, title="", x_label="Iterations", y_label="L_2", names=names, file_name="l2")
plot_curves(x, y4_mean, y4_std, title="", x_label="Iterations", y_label="L_INF", names=names, file_name="linf")
plot_curves(x, y5_mean, y5_std, title="", x_label="Iterations", y_label="Softmax Error", names=names, file_name="sft")