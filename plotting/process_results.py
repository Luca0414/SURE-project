"""
This module draws the figures and performs statistical analysis of the results.
"""

import os

from glob import glob
import json
import pandas as pd
import matplotlib.pyplot as plt

from grouped_boxplot import plot_grouped_boxplot, compute_stats
from constants import RED, BLUE, GREEN, BASELINE_LR, BASELINE_GP, GP_LR

plt.style.use("ggplot")
plt.rcParams.update({"font.size": 22})

if not os.path.exists("figures"):
    os.mkdir("figures")

P_ALPHA = 0.05

df = []

for file in glob("results/*.json"):
    with open(file) as f:
        log = json.load(f)
    for result in log["results"]:
        result["raw_target_expression"] = log["raw_target_expression"]
        result["simplified_target_expression"] = log["simplified_target_expression"]
        result["num_vars"] = log["num_vars"]
        result["seed"] = log["seed"]
    df += log["results"]

df = pd.DataFrame(df)
data_points_worse = df.loc[df["gp_seed_predictive_nrmse"] < df["gp_lr_predictive_nrmse"]]
print("gp_lr_predictive_nrmse was worse in", len(data_points_worse), "cases out of", len(df))
print(data_points_worse[["num_vars", "epsilon"]].value_counts().sort_index())


# RQ1: Number of variables
print("RQ1: Number of variables")
lr_nrmse = list(df.groupby("num_vars")["lr_nrmse"].apply(list))
gp_nrmse = list(df.groupby("num_vars")["gp_seed_nrmse"].apply(list))
gp_lr_nrmse = list(df.groupby("num_vars")["gp_lr_nrmse"].apply(list))

plot_grouped_boxplot(
    [
        [lr_nrmse[0], lr_nrmse[4], lr_nrmse[9]],
        [gp_nrmse[0], gp_nrmse[4], gp_nrmse[9]],
        [gp_lr_nrmse[0], gp_lr_nrmse[4], gp_lr_nrmse[9]],
    ],
    savepath="figures/random_num_vars_nrmse.pgf",
    width=0.6,
    labels=[BASELINE_LR, BASELINE_GP, GP_LR],
    colours=[RED, BLUE, GREEN],
    markers=["x", "o", "+"],
    xticklabels=[1, 5, 10],
    xlabel="Number of variables",
    ylabel="NRMSE",
    yticks=[x / 10 for x in range(0, 11, 2)],
    # Need to hide fliers because LR errors are stupidly large
    showfliers=False,
)
compute_stats(df, "num_vars", P_ALPHA, "figures/random_num_vars_nrmse.csv", outcome="nrmse")

lr_nrmse = list(df.groupby("num_vars")["lr_predictive_nrmse"].apply(list))
gp_nrmse = list(df.groupby("num_vars")["gp_seed_predictive_nrmse"].apply(list))
gp_lr_nrmse = list(df.groupby("num_vars")["gp_lr_predictive_nrmse"].apply(list))

plot_grouped_boxplot(
    [
        [lr_nrmse[0], lr_nrmse[4], lr_nrmse[9]],
        [gp_nrmse[0], gp_nrmse[4], gp_nrmse[9]],
        [gp_lr_nrmse[0], gp_lr_nrmse[4], gp_lr_nrmse[9]],
    ],
    savepath="figures/random_num_vars_predictive_nrmse.pgf",
    width=0.6,
    labels=[BASELINE_LR, BASELINE_GP, GP_LR],
    colours=[RED, BLUE, GREEN],
    markers=["x", "o", "+"],
    xticklabels=[1, 5, 10],
    xlabel="Number of variables",
    ylabel="NRMSE",
    yticks=[x / 10 for x in range(0, 11, 2)],
    # Need to hide fliers because LR errors are stupidly large
    showfliers=False,
)
compute_stats(df, "num_vars", P_ALPHA, "figures/random_num_vars_predictive_nrmse.csv", outcome="predictive_nrmse")


# RQ2: Amount of data
print("\nRQ2: Amount of data")
lr_nrmse = list(df.groupby("data_size")["lr_nrmse"].apply(list))
gp_nrmse = list(df.groupby("data_size")["gp_seed_nrmse"].apply(list))
gp_lr_nrmse = list(df.groupby("data_size")["gp_lr_nrmse"].apply(list))

plot_grouped_boxplot(
    [lr_nrmse, gp_nrmse, gp_lr_nrmse],
    savepath="figures/random_data_nrmse.pgf",
    width=0.6,
    labels=[BASELINE_LR, BASELINE_GP, GP_LR],
    colours=[RED, BLUE, GREEN],
    markers=["x", "o", "+"],
    xticklabels=[10, 50, 100, 500, 1000],
    xlabel="Number of data points",
    ylabel="NRMSE",
    yticks=[x / 10 for x in range(0, 11, 2)],
    # Need to hide fliers because LR errors are stupidly large
    showfliers=False,
)
compute_stats(df, "data_size", P_ALPHA, "figures/random_data_nrmse.csv")
lr_nrmse = list(df.groupby("data_size")["lr_predictive_nrmse"].apply(list))
gp_nrmse = list(df.groupby("data_size")["gp_seed_predictive_nrmse"].apply(list))
gp_lr_nrmse = list(df.groupby("data_size")["gp_lr_predictive_nrmse"].apply(list))

plot_grouped_boxplot(
    [lr_nrmse, gp_nrmse, gp_lr_nrmse],
    savepath="figures/random_data_predictive_nrmse.pgf",
    width=0.6,
    labels=[BASELINE_LR, BASELINE_GP, GP_LR],
    colours=[RED, BLUE, GREEN],
    markers=["x", "o", "+"],
    xticklabels=[10, 50, 100, 500, 1000],
    xlabel="Number of data points",
    ylabel="NRMSE",
    yticks=[x / 10 for x in range(0, 11, 2)],
    # Need to hide fliers because LR errors are stupidly large
    showfliers=False,
)
compute_stats(df, "data_size", P_ALPHA, "figures/random_data_predictive_nrmse.csv")

# RQ3: Amount of noise
print("\nRQ3: Amount of noise")
lr_nrmse = list(df.groupby("epsilon")["lr_nrmse"].apply(list))
gp_nrmse = list(df.groupby("epsilon")["gp_seed_nrmse"].apply(list))
gp_lr_nrmse = list(df.groupby("epsilon")["gp_lr_nrmse"].apply(list))

plot_grouped_boxplot(
    [lr_nrmse, gp_nrmse, gp_lr_nrmse],
    savepath="figures/random_epsilon_nrmse.pgf",
    width=0.6,
    labels=[BASELINE_LR, BASELINE_GP, GP_LR],
    colours=[RED, BLUE, GREEN],
    markers=["x", "o", "+"],
    xticklabels=[0, 0.1, 0.25],
    xlabel="Epsilon",
    ylabel="NRMSE",
    yticks=[x / 10 for x in range(0, 11, 2)],
    # Need to hide fliers because a couple of LR errors are stupidly large
    showfliers=False,
)
compute_stats(df, "epsilon", P_ALPHA, "figures/random_epsilon_nrmse.csv")

lr_nrmse = list(df.groupby("epsilon")["lr_predictive_nrmse"].apply(list))
gp_nrmse = list(df.groupby("epsilon")["gp_seed_predictive_nrmse"].apply(list))
gp_lr_nrmse = list(df.groupby("epsilon")["gp_lr_predictive_nrmse"].apply(list))

plot_grouped_boxplot(
    [lr_nrmse, gp_nrmse, gp_lr_nrmse],
    savepath="figures/random_epsilon_predictive_nrmse.pgf",
    width=0.6,
    labels=[BASELINE_LR, BASELINE_GP, GP_LR],
    colours=[RED, BLUE, GREEN],
    markers=["x", "o", "+"],
    xticklabels=[0, 0.1, 0.25],
    xlabel="Epsilon",
    ylabel="NRMSE",
    yticks=[x / 10 for x in range(0, 11, 2)],
    # Need to hide fliers because a couple of LR errors are stupidly large
    showfliers=False,
)
compute_stats(df, "epsilon", P_ALPHA, "figures/random_epsilon_predictive_nrmse.csv")

# Runtime
print("\nRuntime")
lr_time = list(df.groupby("num_vars")["lr_time"].apply(list))
gp_time = list(df.groupby("num_vars")["gp_seed_time"].apply(list))
gp_lr_time = list(df.groupby("num_vars")["gp_lr_time"].apply(list))

plot_grouped_boxplot(
    [lr_time, gp_time, gp_lr_time],
    savepath="figures/runtime.pgf",
    width=0.6,
    labels=[BASELINE_LR, BASELINE_GP, GP_LR],
    colours=[RED, BLUE, GREEN],
    markers=["x", "o", "+"],
    xticklabels=range(1, 11),
    xlabel="Number of variables",
    ylabel="Runtime (seconds)",
)
compute_stats(df, "num_vars", P_ALPHA, "figures/random_runtime.csv", outcome="time")
