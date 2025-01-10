"""
This module draws the figures and performs statistical analysis of the results.
"""

import os

from glob import glob
import json
import pandas as pd
import matplotlib.pyplot as plt

from grouped_boxplot import plot_grouped_boxplot, compute_stats
from constants import RED, BLUE, GREEN, MAGENTA, BASELINE_LR, BASELINE_GP, GP_LR, ORIGINAL

plt.style.use("ggplot")

if not os.path.exists("figures"):
    os.mkdir("figures")

P_ALPHA = 0.05

df = []
for file in glob("ctf_paper_results/*.json"):
    with open(file) as f:
        log = json.load(f)
        log["file"] = os.path.basename(file).replace(".json", "")
        system = log["file"].split("_")[:-2]
        system[0] = system[0].capitalize()
        if len(system) > 1:
            system[-1] = f"${'_'.join(system[-1].capitalize())}$"
        log["system"] = " ".join(system)
    df.append(log)
df = pd.DataFrame(df)
print(df)

# RQ1: Number of variables
print("NRMSE")
original_nrmse = list(df.groupby("system")["original_model_nrmse"].apply(list))
lr_nrmse = list(df.groupby("system")["lr_nrmse"].apply(list))
gp_nrmse = list(df.groupby("system")["gp_seed_nrmse"].apply(list))
gp_lr_nrmse = list(df.groupby("system")["gp_lr_nrmse"].apply(list))

plot_grouped_boxplot(
    [lr_nrmse, gp_nrmse, gp_lr_nrmse, original_nrmse],
    savepath="figures/ctf_nrmse.png",
    width=0.6,
    labels=[BASELINE_LR, BASELINE_GP, GP_LR, ORIGINAL],
    colours=[RED, BLUE, GREEN, MAGENTA],
    markers=["x", "o", "+", 2],
    xticklabels=df.groupby("system").groups,
    xlabel="Number of variables",
    ylabel="NRMSE",
)
compute_stats(df, "num_vars", P_ALPHA, "figures/ctf_nrmse.csv")


# Runtime
print("\nRuntime")
original_time = list(df.groupby("system")["original_model_nrmse"].apply(list))
lr_time = list(df.groupby("system")["lr_time"].apply(list))
gp_time = list(df.groupby("system")["gp_seed_time"].apply(list))
gp_lr_time = list(df.groupby("system")["gp_lr_time"].apply(list))

plot_grouped_boxplot(
    [lr_time, gp_time, gp_lr_time, original_time],
    savepath="figures/ctf_runtime.png",
    width=0.6,
    labels=[BASELINE_LR, BASELINE_GP, GP_LR, ORIGINAL],
    colours=[RED, BLUE, GREEN, MAGENTA],
    markers=["x", "o", "+", 2],
    xticklabels=df.groupby("system").groups,
    xlabel="Number of variables",
    ylabel="Runtime (seconds)",
)
compute_stats(df, "num_vars", P_ALPHA, "figures/ctf_runtime.csv", outcome="time")
