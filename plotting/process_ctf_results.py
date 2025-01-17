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
plt.rcParams.update({"font.size": 22})

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
    savepath="figures/ctf_nrmse.pgf",
    width=0.6,
    labels=[BASELINE_LR, BASELINE_GP, GP_LR, ORIGINAL],
    colours=[RED, BLUE, GREEN, MAGENTA],
    markers=["x", "o", "+", 2],
    xticklabels=df.groupby("system").groups,
    xlabel="System",
    ylabel="NRMSE",
    yticks=[x / 100 for x in range(0, 7)],
)
compute_stats(df, "system", P_ALPHA, "figures/ctf_nrmse.csv")


# Runtime
print("\nRuntime")
plot_grouped_boxplot(
    [
        list(df.groupby("system")[f"{techique}_time"].apply(list))
        for techique in ["lr", "gp_seed", "gp_lr", "original_model"]
    ],
    savepath="figures/ctf_runtime.pgf",
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
fig, axs = plt.subplots(1, 2, figsize=(8, 6))
axs = axs.reshape(-1)
label = True
for system, ax in zip(df.groupby("system").groups, axs):
    plot_grouped_boxplot(
        [[df.loc[df["system"] == system, f"{techique}_test_nrmse"]] for techique in ["lr", "gp_seed", "gp_lr"]],
        ax=ax,
        width=0.6,
        labels=[BASELINE_LR, BASELINE_GP, GP_LR] if label else None,
        colours=[RED, BLUE, GREEN],
        markers=["x", "o", "+"],
        xticklabels=[system],
        ylabel="Causal Effect Estimate NRSME" if label else None,
        showfliers=False,
        offset=0.05,
        legend=False,
    )
    label = False
axs[0].set_yticks(axs[1].get_yticks() / 100)
# axs[0].set_yticklabels(["-1", "0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"])
axs[0].set_ylim([x / 100 for x in axs[1].get_ylim()])
axs[0].set_xlabel("System")
axs[0].xaxis.set_label_coords(1.05, -0.1)
axs[0].legend(loc="upper center")
plt.savefig("figures/ctf_test_nrmse.pgf", bbox_inches="tight", pad_inches=0)
compute_stats(df, "system", P_ALPHA, "figures/ctf_test_nrmse.csv", outcome="test_nrmse")

print("\nCausal Test Outcomes")
plot_grouped_boxplot(
    [list(df.groupby("system")[f"{techique}_test_bcr"].apply(list)) for techique in ["lr", "gp_seed", "gp_lr"]],
    savepath="figures/ctf_test_bcr.pgf",
    width=0.6,
    labels=[BASELINE_LR, BASELINE_GP, GP_LR],
    colours=[RED, BLUE, GREEN],
    markers=["x", "o", "+"],
    xticklabels=df.groupby("system").groups,
    xlabel="System",
    ylabel="Test Outcome BCR",
    yticks=[x / 10 for x in range(11)],
)
compute_stats(df, "system", P_ALPHA, "figures/ctf_test_bcr.csv", outcome="test_bcr")
