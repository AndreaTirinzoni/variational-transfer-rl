import utils
import numpy as np
import matplotlib.pyplot as plt

MARKERS = ["o", "D", "s", "^", "v", "p", "*"]
COLORS = ["#0e5ad3","#bc2d14", "#a011a3","#d67413","#22aa16", "#d1ba0e" ,"#14ccc2"]
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
    plt.ylim([0,0.85])

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


files = ["results/gvt_gw10x10_like", "results/mgvt_gw10x10_like"]
names = ["GVT - .5k steps", "1-MGVT - .5k steps", "GVT - 1k steps", "1-MGVT - 1k steps",
         "GVT - 4k steps", "1-MGVT - 4k steps"]

iters = 500
pos = int(iters / 50)
rew_pos = 3 # Eval rew (4) or Learning Rew (3)

x1 = []
y1_mean = []
y1_std = []

for file in files:

    x = []
    y_mean = []
    y_std = []
    f = utils.load_object(file)

    for i in range(len(f)):
        r = f[i]
        r2 = f[-i-1]
        if r[0] <= 5.0 and (r[0]*2 % 2 == 0):
            print(r[0])
            x.append(r[0])
            results = r[1]
            results = [a[2] for a in results]
            rews = []
            for result in results:
                rews.append(result[rew_pos][pos])

            results = r2[1]
            results = [a[2] for a in results]
            for result in results:
                rews.append(result[rew_pos][pos])

            if r[0] == 5.0:
                print(rews)

            #print(len(rews))
            y_mean.append(np.mean(rews))
            y_std.append(np.std(rews) / np.sqrt(len(rews)))

    x = np.array(x) / 5.0
    y_mean = np.array(y_mean)
    y_std = np.array(y_std)
    x1.append(x)
    y1_mean.append(y_mean)
    y1_std.append(y_std)

iters = 1000
pos = int(iters / 50)

for file in files:

    x = []
    y_mean = []
    y_std = []
    f = utils.load_object(file)

    for i in range(len(f)):
        r = f[i]
        r2 = f[-i-1]
        if r[0] <= 5.0 and (r[0]*2 % 2 == 0):
            x.append(r[0])
            results = r[1]
            results = [a[2] for a in results]
            rews = []
            for result in results:
                rews.append(result[rew_pos][pos])

            results = r2[1]
            results = [a[2] for a in results]
            for result in results:
                rews.append(result[rew_pos][pos])

            if r[0] == 5.0:
                print(rews)

            #print(len(rews))
            y_mean.append(np.mean(rews))
            y_std.append(np.std(rews) / np.sqrt(len(rews)))

    x = np.array(x) / 5.0
    y_mean = np.array(y_mean)
    y_std = np.array(y_std)
    x1.append(x)
    y1_mean.append(y_mean)
    y1_std.append(y_std)

iters = 4000
pos = int(iters / 50)

for file in files:

    x = []
    y_mean = []
    y_std = []
    f = utils.load_object(file)

    for i in range(len(f)):
        r = f[i]
        r2 = f[-i-1]
        if r[0] <= 5.0 and (r[0]*2 % 2 == 0):
            x.append(r[0])
            results = r[1]
            results = [a[2] for a in results]
            rews = []
            for result in results:
                rews.append(result[rew_pos][pos])

            results = r2[1]
            results = [a[2] for a in results]
            for result in results:
                rews.append(result[rew_pos][pos])

            if r[0] == 5.0:
                print(rews)

            #print(len(rews))
            y_mean.append(np.mean(rews))
            y_std.append(np.std(rews) / np.sqrt(len(rews)))

    x = np.array(x) / 5.0
    y_mean = np.array(y_mean)
    y_std = np.array(y_std)
    x1.append(x)
    y1_mean.append(y_mean)
    y1_std.append(y_std)

plot_curves(x1, y1_mean, y1_std, title="", x_label="Normalized Task Likelihood", y_label="Expected Return", names=names, file_name="lrew")