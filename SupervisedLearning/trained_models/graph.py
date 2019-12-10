from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import numpy as np
import os
import itertools


def load_data(cur_path, exp_name, data_name, use_cols):
    fname = "random_forest\\" + exp_name + "\\" + exp_name + "_" + data_name
    fname = os.path.join(cur_path, fname)
    data = np.loadtxt(fname, delimiter=",", skiprows=1, usecols=use_cols)
    return data

def plot_data(vals, labels, title):
    pass


# Graph trained data results
if __name__ == "__main__":
    exp_name = "rf_test_3"
    data_name1 = "trees_vs_xentropy.txt"
    cur_path = os.path.dirname(__file__)
    use_cols = (0,1,2)
    data = load_data(cur_path, exp_name, data_name1, use_cols)
    x = data[:,0]
    y = data[:,2]
    seed = data[:,1]
    colors = cm.hsv(np.linspace(0, 1, max(seed) + 1))
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.Normalize(vmin=0, vmax=20)
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    for xs, ys, sd in zip(x, y, seed):
        sc = ax1.scatter(xs, ys, color=colors[int(sd)], alpha=0.8)
        # plt.scatter(xs, ys, color=colors[int(sd)], alpha=0.8)
        # plt.colorbar(mappable=cm.ScalarMappable(cmap=colors), ticks=range(20), label='Seed')
        # plt.clim(-0.5, 20.5)
    ax1.set_xlabel('Trees')
    ax1.set_ylabel('Cross Entropy')
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                 norm=norm,
                 orientation='vertical')
    cb.set_label("Seed")
    # plt.colorbar(sc)
    # fig.show()
    plt.suptitle("Cross Entropy vs Number of Trees")
    plt.show()
    # plt.colorbar()
