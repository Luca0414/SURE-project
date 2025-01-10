"""
This module uses gp to fit equations to the data used in the paper
"Testing Causality in Scientific Modelling Software" (https://doi.org/10.1145/3607184).
"""

from time import time
import os
from functools import reduce

import json
import pandas as pd
import statsmodels.formula.api as smf
import deap

from numpy import log, sqrt, sin, cos, tan, power

from causal_testing.estimation.genetic_programming_regression_fitter import GP
from expression_generator import root, reciprocal


def time_execution(fun):
    start_time = time()
    result = fun()
    return result, time() - start_time


def calculate_nrmse(y_estimates, df_outcome):
    sqerrors = (df_outcome - y_estimates) ** 2
    nrmse = sqrt(sqerrors.sum() / len(df_outcome)) / (df_outcome.max() - df_outcome.min())

    if pd.isnull(nrmse) or nrmse.real != nrmse or y_estimates.dtype != df_outcome.dtype:
        return float("inf")

    return nrmse


def pretty_print_ols(model):
    params_dict = model.params.to_dict()
    formula_str = ""
    for variable, coef in params_dict.items():
        if variable == "Intercept":  # Intercept term
            formula_str += f"{coef:.2f}"
        else:
            if coef >= 0:
                formula_str += f" + {coef:.2f}*{variable}"
            else:
                formula_str += f" - {abs(coef):.2f}*{variable}"

    # Remove the trailing " + " if it exists
    formula_str = formula_str.lstrip(" + ")

    # Print the formula with coefficients
    return formula_str


def gp_fit(df, features, outcome, seed, original_ols_formula):
    original_model, original_model_time = time_execution(
        lambda: smf.ols(f"{outcome} ~ {original_ols_formula}", df).fit()
    )

    gp = GP(
        df=df,
        features=features,
        outcome=outcome,
        max_order=3,
        extra_operators=[
            (root, 1),
            (reciprocal, 1),
            (sin, 1),
            (cos, 1),
            (tan, 1),
            (log, 1),
        ],
        sympy_conversions={
            "log": lambda x: f"Log({x},-1)",
            "root": lambda x: f"sqrt({x})",
        },
        seed=seed,
    )
    model, lr_time = time_execution(lambda: smf.ols(f"{outcome} ~ {'+'.join(features)}", df).fit())

    terms = [f"mul({model.params[feature]}, {feature})" for feature in features]
    lr_formula = reduce(lambda acc, term: f"add({acc}, {term})", terms, model.params["Intercept"])
    gp_lr_result, gp_lr_time = time_execution(lambda: gp.run_gp(ngen=100, seeds=[lr_formula]))
    gp_seed_result, gp_seed_time = time_execution(lambda: gp.run_gp(ngen=100, seeds=[lr_formula], repair=False))

    return {
        "data_size": len(df),
        "pset": sorted([k for k, v in gp.pset.mapping.items() if isinstance(v, deap.gp.Primitive)]),
        "num_vars": len(features),
        # Linear regression results
        "lr_raw_formula": str(lr_formula),
        "lr_simplified_formula": str(gp.simplify(lr_formula)),
        "lr_nrmse": gp.fitness(lr_formula)[0],
        "lr_time": lr_time,
        # Our GP results
        "gp_lr_raw_formula": str(gp_lr_result),
        "gp_lr_simplified_formula": str(gp.simplify(gp_lr_result)),
        "gp_lr_nrmse": gp.fitness(gp_lr_result)[0],
        "gp_lr_time": gp_lr_time,
        # Baseline GP (with seed) results
        "gp_seed_raw_formula": str(gp_seed_result),
        "gp_seed_simplified_formula": str(gp.simplify(gp_seed_result)),
        "gp_seed_nrmse": gp.fitness(gp_seed_result)[0],
        "gp_seed_time": gp_seed_time,
        # NRMSE of original model
        "original_model_formula": pretty_print_ols(original_model),
        "original_model_nrmse": calculate_nrmse(original_model.predict(df), df[outcome]),
        "original_model_time": original_model_time,
    }


if not os.path.exists("ctf_paper_results"):
    os.mkdir("ctf_paper_results")

for seed in range(30):
    # Poisson
    # L_t ≈ 2i(w + h)
    lt_result = gp_fit(
        pd.read_csv("ctf_data/poisson_data.csv").astype(float),
        ["width", "height", "intensity"],
        "num_lines_abs",
        seed,
        "I(intensity * (width + height)) - 1",
    )
    with open(f"ctf_paper_results/poisson_lt_result_s{seed}.json", "w") as f:
        json.dump(lt_result, f)

    # Pt ≈ πi^2wh
    pt_result = gp_fit(
        pd.read_csv("ctf_data/poisson_data.csv").astype(float),
        ["width", "height", "intensity"],
        "num_shapes_abs",
        seed,
        "I(intensity * intensity * width * height)",
    )
    with open(f"ctf_paper_results/poisson_pt_results_s{seed}.json", "w") as f:
        json.dump(pt_result, f)

    # Covasim
    covasim_result = gp_fit(
        # "beta" is a special function in sympy, so rename it "β"
        pd.read_csv("ctf_data/covasim_data.csv").rename({"beta": "β"}, axis=1),
        ["β", "avg_rel_sus", "avg_contacts_s", "avg_contacts_h", "avg_contacts_w"],
        "cum_infections",
        seed,
        """
        β + bs(β, degree=3, df=5) +
        log(avg_rel_sus) + power(log(avg_rel_sus), 2) +
        log(total_contacts_w) + power(log(total_contacts_w), 2) +
        log(total_contacts_s) + power(log(total_contacts_s), 2) +
        log(total_contacts_h) + power(log(total_contacts_h), 2) +
        β:log(total_contacts_w) +
        β:log(total_contacts_s) +
        β:log(total_contacts_h) +
        β:log(avg_rel_sus)
        """,
    )
    with open(f"ctf_paper_results/covasim_results_s{seed}.json", "w") as f:
        json.dump(covasim_result, f)
