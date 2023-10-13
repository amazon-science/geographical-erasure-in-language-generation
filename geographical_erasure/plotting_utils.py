# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def draw_comparison_table(
    data1,
    data2,
    top_k=50,
    reverse=True,
    in_percentage=True,
    headers=["Facts", "LM"],
    figsize=(5, 20),
):
    """
    Draws a figure comparing two distributions.

    Parameters
    ----------
        data1: dictionary containing distribution 1 (e.g., key: country, val: population)
        data2: dictionary containing distribution 2
        top_k: to include top-k keys in the distribution from distribution 1
        reverse: whether to reverse sort or not
        in_percentage: whether to display distributions in percentages
        headers: the names of two distributions
        figsize: tuple (x, y) containing figure size
    Returns
    -------
    None
    """

    # Figure Size
    fig, ax = plt.subplots(figsize=figsize)

    # sort the first data dictionary and sort it to get top-k
    sorted_list = sorted(data1.items(), key=lambda x: x[1], reverse=reverse)[:top_k]

    x = [i[0] for i in sorted_list]

    y1 = [data1[key] for key in x]
    y2 = [data2[key] for key in x]

    if in_percentage:
        y1 = [i * 100.0 for i in y1]
        y2 = [i * 100.0 for i in y2]

    assert len(x) == len(y1)
    assert len(x) == len(y2)
    X_axis = np.arange(len(x))
    width = 0.1

    # Plot the data in horizontal bars
    ax.barh(X_axis, y1, label=headers[0], alpha=0.5)
    ax.barh(X_axis, y2, label=headers[1], alpha=0.5)

    # Remove axes splines
    for s in ["top", "bottom", "left", "right"]:
        ax.spines[s].set_visible(False)

    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)

    # set the X axis (e.g., country names)
    plt.yticks(X_axis, x)

    # Add x, y gridlines
    ax.grid(color="grey", linestyle="-.", linewidth=0.25)
    # alpha = 0.2)

    # Show top values
    ax.invert_yaxis()

    # add legend
    plt.legend()

    # Show Plot
    plt.show()

    return


def draw_comparison_table_multiple(
    ground_truth,
    data,
    top_k=50,
    reverse=True,
    in_percentage=True,
    figsize=(5, 20),
    savepath=None,
    xlim=None,
    erasure_set=None,
):
    """
    Draws a figure comparing multiple predicted distributions against ground truth, as boxplots.

    Parameters
    ----------
    ground_truth: dictionary containing distribution 1 (e.g., key: country, val: population)
    data: list of dictionaries containing all the predicted distributions
    top_k: to include top-k keys in the distribution from distribution 1
    reverse: whether to reverse sort or not
    in_percentage: whether to display distributions in percentages
    figsize: tuple (x, y) containing figure size
    savepath: str, where to save the resulting figure (if None, fig. is not saved)
    xlim: x axis limits, passed to matplotlib
    erasure_set: if an erasure set, i.e. a list of countries is passed, country names are typeset in red

    Returns
    -------
    None
    """

    fig, ax = plt.subplots(figsize=figsize)

    # sort the first data dictionary and sort it to get top-k
    sorted_list = sorted(ground_truth.items(), key=lambda x: x[1], reverse=reverse)[
        :top_k
    ]

    x = [i[0] for i in sorted_list]

    ground_truth = [ground_truth[key] for key in x]
    y = np.empty((len(data), len(x)))

    if in_percentage:
        norm_c = 100.0
    else:
        norm_c = 1
    for prediction_ix in range(len(data)):
        y[prediction_ix] = [data[prediction_ix][key] * norm_c for key in x]
    ground_truth = np.array(ground_truth) * norm_c

    X_axis = np.arange(len(x))
    width = 0.1

    # Plot the data in horizontal bar-/boxplot
    p1 = ax.barh(X_axis, ground_truth, label="ground truth", alpha=0.3, color="k")
    p2 = sns.boxplot(y, orient="h", showfliers=False)

    # Remove axes splines
    for s in ["top", "bottom", "left", "right"]:
        ax.spines[s].set_visible(False)

    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)

    if xlim is not None:
        ax.set_xlim(xlim)

    # set the X axis (e.g., country names)
    plt.yticks(X_axis, x, fontsize=20)

    # color country names red if in erasure set
    colors = ["darkred" if c in erasure_set else "black" for c in x]
    for ytick, color in zip(ax.get_yticklabels(), colors):
        ytick.set_color(color)

    ax.set_xlabel(r"$p(x_i|c)$", fontsize=20)

    # if xlim is not None:
    xvals = np.linspace(0, xlim[-1], 5 + 1)
    ax.set_xticks(xvals, [f"{int(value)}%" for value in xvals], fontsize=20)

    # Add x, y gridlines
    ax.grid(color="white", linestyle="-", linewidth=1)

    # add legend
    ax.legend(fontsize=20)

    p2 = plt.Rectangle((0, 0), 1, 1, fc="gray")
    ax.legend([p2], ["Ground Truth"], fontsize=20)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight", pad_inches=0.1)

    # Show Plot
    plt.show()

    return
