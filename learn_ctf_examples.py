"""
This module uses gp to fit equations to the data used in the paper
"Testing Causality in Scientific Modelling Software" (https://doi.org/10.1145/3607184).
"""

import os
import argparse
from time import time
from functools import reduce

import json
import deap
import pandas as pd
import statsmodels.formula.api as smf
from numpy import log, sin, cos, tan, power
from sklearn.metrics import balanced_accuracy_score

from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_outcome import ExactValue
from causal_testing.estimation.genetic_programming_regression_fitter import GP
from causal_testing.estimation.abstract_estimator import Estimator

from expression_generator import root, reciprocal
from learn_equations import calculate_nrmse


class GPFormulaEstimator(Estimator):
    def __init__(
        self,
        pset,
        df,
        formula,
        adjustment_set_config,
        treatment_variable,
        control_value,
        treatment_value,
        outcome_variable,
    ):
        super().__init__(
            treatment=treatment_variable,
            treatment_value=treatment_value,
            control_value=control_value,
            adjustment_set=adjustment_set_config,
            outcome=outcome_variable,
            df=df,
        )
        self.pset = pset
        self.formula = formula
        self.adjustment_set_config = adjustment_set_config

    def add_modelling_assumptions(self):
        pass

    def _estimate(self):
        if hasattr(self.formula, "get_prediction"):
            x = pd.DataFrame(columns=self.df.columns)
            x["Intercept"] = 1  # self.intercept
            x[self.treatment] = [self.treatment_value, self.control_value]
            for k, v in self.adjustment_set_config.items():
                x[k] = v
            prediction = self.formula.get_prediction(x).summary_frame()
            control_outcome, treatment_outcome = prediction.iloc[1]["mean"], prediction.iloc[0]["mean"]
        else:
            func = deap.gp.compile(deap.gp.PrimitiveTree.from_string(str(self.formula), self.pset), self.pset)
            control_outcome = func(**(self.adjustment_set_config | {self.treatment: self.control_value}))
            treatment_outcome = func(**(self.adjustment_set_config | {self.treatment: self.treatment_value}))
        return control_outcome, treatment_outcome

    def estimate_ate(self):
        control_outcome, treatment_outcome = self._estimate()
        return [treatment_outcome - control_outcome], (None, None)

    def estimate_risk_ratio(self):
        control_outcome, treatment_outcome = self._estimate()
        return [treatment_outcome / control_outcome], (None, None)


def time_execution(fun):
    start_time = time()
    result = fun()
    return result, time() - start_time


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


def gp_fit(df, features, outcome, seed, original_ols_formula, run_ctf):
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

    original_test_outcomes = run_ctf(df=df, formula=original_model, pset=gp.pset)
    original_effect_estimates = pd.Series([test["effect_estimate"] for test in original_test_outcomes])
    original_test_results = pd.Series([test["passed"] for test in original_test_outcomes])

    lr_test_outcomes = run_ctf(df=df, formula=model, pset=gp.pset)
    gp_lr_test_outcomes = run_ctf(df=df, formula=gp_lr_result, pset=gp.pset)
    gp_seed_test_outcomes = run_ctf(df=df, formula=gp_seed_result, pset=gp.pset)

    return {
        "data_size": len(df),
        "pset": sorted([k for k, v in gp.pset.mapping.items() if isinstance(v, deap.gp.Primitive)]),
        "num_vars": len(features),
        # Linear regression results
        "lr_raw_formula": str(lr_formula),
        "lr_simplified_formula": str(gp.simplify(lr_formula)),
        "lr_nrmse": gp.fitness(lr_formula)[0],
        "lr_test_outcomes": lr_test_outcomes,
        "lr_test_nrmse": calculate_nrmse(
            pd.Series([test["effect_estimate"] for test in lr_test_outcomes]), original_effect_estimates
        ),
        "lr_test_bcr": balanced_accuracy_score(
            pd.Series([test["passed"] for test in lr_test_outcomes]), original_test_results
        ),
        "lr_time": lr_time,
        # Our GP results
        "gp_lr_raw_formula": str(gp_lr_result),
        "gp_lr_simplified_formula": str(gp.simplify(gp_lr_result)),
        "gp_lr_nrmse": gp.fitness(gp_lr_result)[0],
        "gp_lr_test_outcomes": gp_lr_test_outcomes,
        "gp_lr_test_nrmse": calculate_nrmse(
            pd.Series([test["effect_estimate"] for test in gp_lr_test_outcomes]), original_effect_estimates
        ),
        "gp_lr_test_bcr": balanced_accuracy_score(
            pd.Series([test["passed"] for test in gp_lr_test_outcomes]), original_test_results
        ),
        "gp_lr_time": gp_lr_time,
        # Baseline GP (with seed) results
        "gp_seed_raw_formula": str(gp_seed_result),
        "gp_seed_simplified_formula": str(gp.simplify(gp_seed_result)),
        "gp_seed_nrmse": gp.fitness(gp_seed_result)[0],
        "gp_seed_test_outcomes": gp_seed_test_outcomes,
        "gp_seed_test_nrmse": calculate_nrmse(
            pd.Series([test["effect_estimate"] for test in gp_seed_test_outcomes]), original_effect_estimates
        ),
        "gp_seed_test_bcr": balanced_accuracy_score(
            pd.Series([test["passed"] for test in gp_seed_test_outcomes]), original_test_results
        ),
        "gp_seed_time": gp_seed_time,
        # NRMSE of original model
        "original_model_formula": pretty_print_ols(original_model),
        "original_model_nrmse": calculate_nrmse(original_model.predict(df), df[outcome]),
        "original_test_outcomes": original_test_outcomes,
        "original_model_time": original_model_time,
    }


def run_causal_test(
    df,
    pset,
    formula,
    treatment_variable,
    control_value,
    treatment_value,
    outcome_variable,
    estimate_type,
    expected_causal_effect,
    adjustment_set_config,
):
    base_test_case = BaseTestCase(
        treatment_variable=treatment_variable,
        outcome_variable=outcome_variable,
        effect="direct",
    )

    causal_test_case = CausalTestCase(
        base_test_case=base_test_case,
        expected_causal_effect=expected_causal_effect,
        control_value=control_value,
        treatment_value=treatment_value,
        estimate_type=estimate_type,
    )
    estimation_model = GPFormulaEstimator(
        pset=pset,
        df=df,
        formula=formula,
        adjustment_set_config=adjustment_set_config,
        treatment_variable=treatment_variable,
        control_value=control_value,
        treatment_value=treatment_value,
        outcome_variable=outcome_variable,
    )
    causal_test_result = causal_test_case.execute_test(estimation_model, None)
    return causal_test_result.to_dict(json=True) | {
        "passed": bool(expected_causal_effect.apply(causal_test_result)),
        "formula": pretty_print_ols(formula) if hasattr(formula, "params") else str(formula),
        "adjustment_set": adjustment_set_config,
        "effect_estimate": causal_test_result.test_value.value[0],
        "expected_causal_effect": str(expected_causal_effect),
    }


def run_ctf_poisson(df, formula, pset):
    return [
        run_causal_test(
            df=df,
            pset=pset,
            formula=formula,
            treatment_variable="intensity",
            control_value=i,
            treatment_value=2 * i,
            outcome_variable="num_shapes_unit",
            estimate_type="risk_ratio",
            expected_causal_effect=ExactValue(4, 0.5),
            adjustment_set_config={"width": size, "height": size},
        )
        for i in [1, 2, 4, 8]
        for size in range(1, 11)
    ]


def run_ctf_covasim(df, formula, pset):
    oracle = pd.read_csv("ctf_example_data/covasim_oracle.csv", index_col=0)
    return [
        run_causal_test(
            df=df,
            pset=pset,
            formula=formula,
            treatment_variable="β",
            control_value=0.016,
            treatment_value=0.02672,
            outcome_variable="cum_infections",
            estimate_type="ate",
            expected_causal_effect=ExactValue(
                oracle["change_in_infections"][location],
                0.055 * oracle["change_in_infections"][location],
            ),
            adjustment_set_config=df.loc[
                df["location"] == location,
                ["avg_rel_sus", "total_contacts_h", "total_contacts_s", "total_contacts_w"],
            ]
            .mean()
            .to_dict(),
        )
        | {
            "adjustment_set": df.loc[
                df["location"] == location,
                ["avg_rel_sus", "total_contacts_h", "total_contacts_s", "total_contacts_w"],
            ]
            .mean()
            .to_dict()
            | {"location": location}
        }
        for location in set(df["location"])
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python learn_equations.py",
        description="Learn the equations from the supplied JSON file.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="The random seed.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        required=True,
        help="The output directory.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    # Poisson
    poisson_data = pd.read_csv("ctf_example_data/poisson_data.csv").astype(float)
    poisson_data["num_shapes_unit"] = poisson_data["num_shapes_abs"] / (poisson_data["width"] * poisson_data["height"])
    original_estimation_equation = "intensity + I(intensity ** 2) - 1"
    pt_result = gp_fit(
        poisson_data,
        ["width", "height", "intensity"],
        "num_shapes_unit",
        args.seed,
        original_estimation_equation,
        run_ctf_poisson,
    )
    with open(f"{args.outdir}/poisson_result_s{args.seed}.json", "w") as f:
        json.dump(pt_result, f)

    # Covasim
    covasim_result = gp_fit(
        # "beta" is a special function in sympy, so rename it "β"
        pd.read_csv("ctf_example_data/covasim_data.csv").rename({"beta": "β"}, axis=1),
        ["β", "avg_rel_sus", "total_contacts_s", "total_contacts_h", "total_contacts_w"],
        "cum_infections",
        args.seed,
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
        run_ctf_covasim,
    )
    with open(f"{args.outdir}/covasim_result_s{args.seed}.json", "w") as f:
        json.dump(covasim_result, f)
