"""
This module provides common code to define a grouped boxplot.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman


def color(colour, flierprops=None):
    if flierprops is None:
        flierprops = {}
    return {
        "boxprops": {"color": colour},
        "capprops": {"color": colour},
        "whiskerprops": {"color": colour},
        "flierprops": {"color": colour, "markeredgecolor": colour} | flierprops,
        "medianprops": {"color": colour},
    }


def plot_grouped_boxplot(
    groups,
    savepath=None,
    width=0.6,
    labels=None,
    colours=None,
    markers=None,
    title=None,
    xticklabels=None,
    xlabel=None,
    ylabel=None,
    ax=None,
    showfliers=True,
    yticks=None,
    offset=0,
    figsize=(10, 6),
    legend=True,
):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    positions = max(len(x) for x in groups)
    plots = len(groups)
    if isinstance(labels, list) and len(labels) != plots:
        raise ValueError(
            f"If providing labels, please ensure that you provide as many as you have plots ({len(labels)})"
        )
    if isinstance(colours, list) and len(colours) != plots:
        raise ValueError("If providing colours, please ensure that you provide as many as you have plots")
    if isinstance(showfliers, bool):
        showfliers = [showfliers] * len(groups)
    if isinstance(showfliers, list) and len(showfliers) != plots:
        raise ValueError("If providing showfliers, please ensure that you provide as many as you have plots")
    for i, boxes in enumerate(groups):
        marker = markers[i] if isinstance(markers, list) else markers if markers is not None else "o"

        ax.boxplot(
            boxes,
            positions=np.array(range(positions)) * (plots + 1) + i + offset,
            widths=width,
            showfliers=showfliers[i],
            label=labels[i] if labels is not None else None,
            **color(
                colours[i] if colours is not None else None,
                flierprops={"marker": marker, "markersize": width * 2},
            ),
        )

    if xticklabels is not None:
        ax.set_xticks(
            np.array(range(len(xticklabels))) * (plots + 1) + (((plots + (plots / 2) - 1) * width) / 2),
            xticklabels,
        )
    if labels is not None and legend:
        ax.legend()
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if yticks is not None:
        ax.set_yticks(yticks)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight", pad_inches=0)
        plt.clf()


def compute_stats(df, groupby, p_alpha, save_path, outcome="nrmse"):
    stats = [
        (gp_inx, friedmanchisquare(group[f"lr_{outcome}"], group[f"gp_seed_{outcome}"], group[f"gp_lr_{outcome}"]))
        for gp_inx, group in df.groupby(groupby)
    ]
    stats_data = []
    for gp_inx, stat in stats:
        datum = {groupby: gp_inx, "friedman_pvalue": stat.pvalue}
        if stat.pvalue < p_alpha:
            techniques = [f"lr_{outcome}", f"gp_seed_{outcome}", f"gp_lr_{outcome}"]
            if f"original_model_{outcome}" in df:
                techniques.append(f"original_model_{outcome}")
            nemenyi = posthoc_nemenyi_friedman(df.loc[df[groupby] == gp_inx, techniques])
            datum["gp_lr/lr"] = nemenyi[f"gp_lr_{outcome}"][f"lr_{outcome}"]
            datum["gp_lr/gp"] = nemenyi[f"gp_lr_{outcome}"][f"gp_seed_{outcome}"]
            if f"original_model_{outcome}" in df:
                datum["gp_lr/original"] = nemenyi[f"gp_lr_{outcome}"][f"original_model_{outcome}"]
        else:
            datum["gp_lr/lr"] = None
            datum["gp_lr/gp"] = None
            if f"original_model_{outcome}" in df:
                datum["gp_lr/original"] = None
        stats_data.append(datum)
    stats_data = pd.DataFrame(stats_data)
    stats_data.to_csv(save_path)
    print(stats_data)
