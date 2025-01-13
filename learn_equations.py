"""
This is the main script for our evaluation.
It generates and infers equations according to the supplied parameters.
"""

import json
import argparse
import pandas as pd
from numpy import sin, cos, tan, log, array, sqrt
import deap
from functools import reduce
import statsmodels.formula.api as smf
from time import time
import os
from stopit import ThreadingTimeout

from causal_testing.estimation.genetic_programming_regression_fitter import GP
from expression_generator import ExpressionGenerator, root, reciprocal


def time_execution(fun):
    start_time = time()
    result = fun()
    return result, time() - start_time


def calculate_nrmse(estimated_outputs, expected_outputs):
    sqerrors = (expected_outputs - estimated_outputs) ** 2
    nrmse = sqrt(sqerrors.sum() / len(expected_outputs)) / (expected_outputs.max() - expected_outputs.min())

    if pd.isnull(nrmse) or nrmse.real != nrmse or estimated_outputs.dtype != expected_outputs.dtype:
        return float("inf")

    return nrmse


def evaluate_predictive_accuracy(expression, pset, inputs, expected_outputs):
    func = deap.gp.compile(deap.gp.PrimitiveTree.from_string(expression, pset), pset)
    estimated_outputs = array([func(**row.to_dict()) for _, row in inputs.iterrows()])
    return calculate_nrmse(estimated_outputs, expected_outputs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="python learn_equations.py",
        description="Learn the equations from the supplied JSON file.",
    )
    parser.add_argument(
        "-n",
        "--num_vars",
        type=int,
        required=True,
        help="The number of variables in an expression.",
    )
    parser.add_argument(
        "-d",
        "--data_size",
        type=int,
        nargs="+",
        required=True,
        help="The number of data points to use for training data.",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=int,
        default=[0],
        help="The percentage noise factor. Defaults to 0.",
        nargs="+",
    )
    parser.add_argument(
        "-E",
        "--expression",
        type=str,
        help="An equation to infer.",
    )
    parser.add_argument(
        "-p",
        "--predictive_data_size",
        type=int,
        default=100,
        help="The number of data points to use to evaluate predictive accuracy.",
    )
    parser.add_argument("-o", "--outfile", help="Where to save the output.")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed.")

    args = parser.parse_args()

    generator = ExpressionGenerator(args.num_vars, args.seed)

    full_data_size = max(args.data_size)

    if args.expression is not None:
        expression = args.expression
    else:
        # Keep trying to generate expressions until we get one that properly evaluates.
        # We may get null values if we have a divide by zero error or something.
        isnull = True
        while isnull:
            expression = generator.generate_expression()
            df, outputs = generator.generate_training_data(expression, full_data_size)
            predictive_df, predictive_outputs = generator.generate_training_data(expression, args.predictive_data_size)
            df["outputs"] = outputs
            isnull = df.isnull().any().any()

    # Seed the population with the standard LR formula
    features = [f"x{i}" for i in range(args.num_vars)]

    results = []

    for epsilon in args.epsilon:
        print("epsilon", epsilon)
        outcome = f"outputs_{epsilon}"
        df[outcome] = generator.add_noise(df["outputs"], epsilon / 100)
        for data_size in args.data_size:
            print("data_size", data_size)
            model, lr_time = time_execution(lambda: smf.ols(f"{outcome} ~ {'+'.join(features)}", df).fit())

            terms = [f"mul({model.params[feature]}, {feature})" for feature in features]
            lr_formula = reduce(lambda acc, term: f"add({acc}, {term})", terms, model.params["Intercept"])

            gp = GP(
                df=df.sample(data_size),
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
                seed=args.seed,
            )

            gp_lr_result, gp_lr_time = time_execution(lambda: gp.run_gp(ngen=100, seeds=[lr_formula]))
            gp_seed_result, gp_seed_time = time_execution(lambda: gp.run_gp(ngen=100, seeds=[lr_formula], repair=False))

            with ThreadingTimeout(10) as lr_timeout_ctx:
                lr_simplified_formula = str(gp.simplify(lr_formula))
            with ThreadingTimeout(10) as gplr_timeout_ctx:
                gp_lr_simplified_formula = str(gp.simplify(gp_lr_result))
            with ThreadingTimeout(10) as gp_seed_timeout_ctx:
                gp_seed_simplified_formula = str(gp.simplify(gp_seed_result))

            results.append(
                {
                    "epsilon": epsilon,
                    "data_size": data_size,
                    "pset": sorted([k for k, v in gp.pset.mapping.items() if isinstance(v, deap.gp.Primitive)]),
                    # Linear regression results
                    "lr_raw_formula": str(lr_formula),
                    "lr_simplified_formula": (
                        lr_simplified_formula if lr_timeout_ctx.state == lr_timeout_ctx.EXECUTED else None
                    ),
                    "lr_nrmse": gp.fitness(lr_formula)[0],
                    "lr_predictive_nrmse": evaluate_predictive_accuracy(
                        str(lr_formula), gp.pset, predictive_df, predictive_outputs
                    ),
                    "lr_time": lr_time,
                    # Our GP results
                    "gp_lr_raw_formula": str(gp_lr_result),
                    "gp_lr_simplified_formula": (
                        gp_lr_simplified_formula if gplr_timeout_ctx.state == gplr_timeout_ctx.EXECUTED else None
                    ),
                    "gp_lr_nrmse": gp.fitness(gp_lr_result)[0],
                    "gp_lr_predictive_nrmse": evaluate_predictive_accuracy(
                        str(gp_lr_result), gp.pset, predictive_df, predictive_outputs
                    ),
                    "gp_lr_time": gp_lr_time,
                    # Baseline GP (with seed) results
                    "gp_seed_raw_formula": str(gp_seed_result),
                    "gp_seed_simplified_formula": (
                        gp_seed_simplified_formula
                        if gp_seed_timeout_ctx.state == gp_seed_timeout_ctx.EXECUTED
                        else None
                    ),
                    "gp_seed_nrmse": gp.fitness(gp_seed_result)[0],
                    "gp_seed_predictive_nrmse": evaluate_predictive_accuracy(
                        str(gp_seed_result), gp.pset, predictive_df, predictive_outputs
                    ),
                    "gp_seed_time": gp_seed_time,
                }
            )

    result = {
        "raw_target_expression": expression,
        "simplified_target_expression": str(gp.simplify(expression)),
        "num_vars": args.num_vars,
        "seed": args.seed,
        "results": results,
    }

    if args.outfile:
        if not os.path.exists(os.path.dirname(args.outfile)):
            os.mkdir(os.path.dirname(args.outfile))
        with open(args.outfile, "w") as f:
            json.dump(result, f)
