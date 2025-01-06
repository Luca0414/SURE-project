import json
import argparse
import pandas as pd
import numpy as np
import deap
from functools import reduce
import statsmodels.formula.api as smf
from time import time
import os

from causal_testing.estimation.genetic_programming_regression_fitter import GP
from expression_generator import ExpressionGenerator, root, reciprocal


def time_execution(fun):
    start_time = time()
    result = fun()
    return result, time() - start_time


parser = argparse.ArgumentParser(
    prog="LearnEquations",
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
parser.add_argument("-o", "--outfile", help="Where to save the output.")
parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed.")

args = parser.parse_args()


generator = ExpressionGenerator(args.num_vars, args.seed)

full_data_size = max(args.data_size)


# Keep trying to generate expressions until we get one that properly evaluates.
# We may get null values if we have a divide by zero error or something.
isnull = True
while isnull:
    expression = generator.generate_expression()
    df, outputs = generator.generate_training_data(expression, full_data_size)
    df["outputs"] = outputs
    isnull = df.isnull().any().any()
    print(expression)
    print(df)

# Seed the population with the standard LR formula
features = [f"x{i}" for i in range(args.num_vars)]

results = []

for epsilon in args.epsilon:
    print("epsilon", epsilon)
    outcome = f"outputs_{epsilon}"
    df[outcome] = generator.add_noise(df["outputs"], epsilon / 100)
    for data_size in args.data_size:
        print("data_size", data_size)
        model, lr_time = time_execution(
            lambda: smf.ols(f"{outcome} ~ {'+'.join(features)}", df).fit()
        )

        terms = [f"mul({model.params[feature]}, {feature})" for feature in features]
        lr_formula = reduce(
            lambda acc, term: f"add({acc}, {term})", terms, model.params["Intercept"]
        )

        gp = GP(
            df=df.sample(data_size),
            features=features,
            outcome=outcome,
            max_order=3,
            extra_operators=[
                (root, 1),
                (reciprocal, 1),
                (np.sin, 1),
                (np.cos, 1),
                (np.tan, 1),
                (np.log, 1),
            ],
            sympy_conversions={
                "log": lambda x: f"Log({x},-1)",
                "root": lambda x: f"sqrt({x})",
            },
            seed=args.seed,
        )

        gp_lr_result, gp_lr_time = time_execution(
            lambda: gp.run_gp(ngen=100, seeds=[lr_formula])
        )
        gp_seed_result, gp_seed_time = time_execution(
            lambda: gp.run_gp(ngen=100, seeds=[lr_formula], repair=False)
        )

        results.append(
            {
                "epsilon": epsilon,
                "data_size": data_size,
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
