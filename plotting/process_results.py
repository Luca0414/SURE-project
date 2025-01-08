import os

from glob import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from grouped_boxplot import plot_grouped_boxplot
from constants import RED, BLUE, GREEN, BASELINE_LR, BASELINE_GP, GP_LR

plt.style.use("ggplot")

if not os.path.exists("figures"):
    os.mkdir("figures")

df = []
problems = []
for file in glob("results/*.json"):
    with open(file) as f:
        try:
            log = json.load(f)
        except json.decoder.JSONDecodeError:
            print(file)
            problems.append(file[8:-5].replace("n", "").replace("s", "").split("_"))
    for result in log["results"]:
        result["raw_target_expression"] = log["raw_target_expression"]
        result["simplified_target_expression"] = log["simplified_target_expression"]
        result["num_vars"] = log["num_vars"]
        result["seed"] = log["seed"]
    df += log["results"]

df = pd.DataFrame(df)

with open("configurations.txt") as f:
    for i, line in enumerate(f, 1):
        for n, s in problems:
            if f"-n {n} " in line and f"-s {s} " in line:
                print(f"sbatch learn_equations.sh configurations.txt {i}")

# RQ1: Number of variables
lr_nrmse = list(df.groupby("num_vars")["lr_nrmse"].apply(list))
gp_nrmse = list(df.groupby("num_vars")["gp_seed_nrmse"].apply(list))
gp_lr_nrmse = list(df.groupby("num_vars")["gp_lr_nrmse"].apply(list))

plot_grouped_boxplot(
    [lr_nrmse, gp_nrmse, gp_lr_nrmse],
    savepath="figures/rq1_num_vars_nrmse.png",
    width=0.6,
    labels=[BASELINE_LR, BASELINE_GP, GP_LR],
    colours=[RED, BLUE, GREEN],
    markers=["x", "o", "+"],
    xticklabels=range(1, 10),
    xlabel="Number of variables",
    ylabel="NRMSE",
    # Need to hide fliers because LR errors are stupidly large
    showfliers=False,
)

# RQ2: Amount of data
lr_nrmse = list(df.groupby("data_size")["lr_nrmse"].apply(list))
gp_nrmse = list(df.groupby("data_size")["gp_seed_nrmse"].apply(list))
gp_lr_nrmse = list(df.groupby("data_size")["gp_lr_nrmse"].apply(list))

plot_grouped_boxplot(
    [lr_nrmse, gp_nrmse, gp_lr_nrmse],
    savepath="figures/rq2_data_nrmse.png",
    width=0.6,
    labels=[BASELINE_LR, BASELINE_GP, GP_LR],
    colours=[RED, BLUE, GREEN],
    markers=["x", "o", "+"],
    xticklabels=[10, 50, 100, 500, 1000],
    xlabel="Number of data points",
    ylabel="NRMSE",
    # Need to hide fliers because LR errors are stupidly large
    showfliers=False,
)

# RQ3: Amount of noise
lr_nrmse = list(df.groupby("epsilon")["lr_nrmse"].apply(list))
gp_nrmse = list(df.groupby("epsilon")["gp_seed_nrmse"].apply(list))
gp_lr_nrmse = list(df.groupby("epsilon")["gp_lr_nrmse"].apply(list))

plot_grouped_boxplot(
    [lr_nrmse, gp_nrmse, gp_lr_nrmse],
    savepath="figures/rq3_epsilon_nrmse.png",
    width=0.6,
    labels=[BASELINE_LR, BASELINE_GP, GP_LR],
    colours=[RED, BLUE, GREEN],
    markers=["x", "o", "+"],
    xticklabels=[0, 0.1, 0.25],
    xlabel="Epsilon",
    ylabel="NRMSE",
    # Need to hide fliers because LR errors are stupidly large
    showfliers=False,
)
