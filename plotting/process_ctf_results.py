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
for file in glob("ctf_example_results/*.json"):
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

# RQ1: Number of variables
print("NRMSE")
plot_grouped_boxplot(
    [
        list(df.groupby("system")[f"{techique}_nrmse"].apply(list))
        for techique in ["lr", "gp_seed", "gp_lr", "original_model"]
    ],
    savepath="figures/ctf_nrmse.png",
    width=0.6,
    labels=[BASELINE_LR, BASELINE_GP, GP_LR, ORIGINAL],
    colours=[RED, BLUE, GREEN, MAGENTA],
    markers=["x", "o", "+", 2],
    xticklabels=df.groupby("system").groups,
    xlabel="System",
    ylabel="NRMSE",
)
compute_stats(df, "system", P_ALPHA, "figures/ctf_nrmse.csv")


# Runtime
print("\nRuntime")
plot_grouped_boxplot(
    [
        list(df.groupby("system")[f"{techique}_time"].apply(list))
        for techique in ["lr", "gp_seed", "gp_lr", "original_model"]
    ],
    savepath="figures/ctf_runtime.png",
    width=0.6,
    labels=[BASELINE_LR, BASELINE_GP, GP_LR, ORIGINAL],
    colours=[RED, BLUE, GREEN, MAGENTA],
    markers=["x", "o", "+", 2],
    xticklabels=df.groupby("system").groups,
    xlabel="System",
    ylabel="Runtime (seconds)",
)
compute_stats(df, "system", P_ALPHA, "figures/ctf_runtime.csv", outcome="time")

# Causal test outcomes
# No result plotted for "original" since this is the gold standard comparison for the other three
print("\nCausal Effect Estimates")
plot_grouped_boxplot(
    [list(df.groupby("system")[f"{techique}_test_nrmse"].apply(list)) for techique in ["lr", "gp_seed", "gp_lr"]],
    savepath="figures/ctf_test_nrmse.png",
    width=0.6,
    labels=[BASELINE_LR, BASELINE_GP, GP_LR],
    colours=[RED, BLUE, GREEN],
    markers=["x", "o", "+"],
    xticklabels=df.groupby("system").groups,
    xlabel="System",
    ylabel="Causal Effect Estimate NRSME",
    showfliers=False,
)
compute_stats(df, "system", P_ALPHA, "figures/ctf_test_nrmse.csv", outcome="test_nrmse")

print("\nCausal Test Outcomes")
plot_grouped_boxplot(
    [list(df.groupby("system")[f"{techique}_test_bcr"].apply(list)) for techique in ["lr", "gp_seed", "gp_lr"]],
    savepath="figures/ctf_test_bcr.png",
    width=0.6,
    labels=[BASELINE_LR, BASELINE_GP, GP_LR],
    colours=[RED, BLUE, GREEN],
    markers=["x", "o", "+"],
    xticklabels=df.groupby("system").groups,
    xlabel="System",
    ylabel="Test Outcome BCR",
)
compute_stats(df, "system", P_ALPHA, "figures/ctf_test_bcr.csv", outcome="test_bcr")
