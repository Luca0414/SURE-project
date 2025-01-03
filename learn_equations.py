from causal_testing.estimation.genetic_programming_regression_fitter import GP
import json
import argparse
from make_linear_equations import root, reciprocal
import pandas as pd
import numpy as np
import deap
from functools import reduce
import statsmodels.formula.api as smf
from time import time


def add_noise(array, noise_factor):
    """
    Adds noise to each element of a numpy array.

    Args:
      array: The input numpy array.
      noise_factor: The percentage of noise to add to each element (between 0 and 1).

    Returns:
      A new numpy array with noise added to each element.
    """
    if noise_factor == 0:
        return array
    noise = np.random.uniform(-1, 1, size=array.size) * noise_factor * array
    return array + noise


parser = argparse.ArgumentParser(
    prog="LearnEquations",
    description="Learn the equations from the supplied JSON file.",
)
parser.add_argument(
    "-i",
    "--index",
    type=int,
    help="The index of the equation to learn, defaults to all.",
)
parser.add_argument(
    "-n",
    "--num_points",
    type=int,
    help="The number of data points to use for training data.",
)
parser.add_argument(
    "-e", "--epsilon", type=float, default=0, help="The noise factor. Defaults to 0."
)
parser.add_argument("-o", "--outfile", help="Where to save the output.")
parser.add_argument("-s", "--seed", default=0, help="Random seed.")
parser.add_argument(
    "equations", help="The JSON file containing the equations and training data."
)

args = parser.parse_args()

with open(args.equations) as f:
    equations = json.load(f)

if args.index is not None:
    equations = [equations[args.index]]

for equation in equations:
    equation["epsilon"] = args.epsilon

    df = pd.DataFrame(equation["training_data"]).reset_index()
    df["outputs"] = add_noise(np.array(equation["outputs"]), args.epsilon)

    if args.num_points is None:
        equation["num_points"] = len(df)
    else:
        equation["num_points"] = args.num_points
        df = df.sample(args.num_points, random_state=args.seed, replace=False)

    # Seed the population with the standard LR formula
    features = [f"x{i}" for i in range(equation["num_vars"])]

    lr_start_time = time()
    model = smf.ols(f"outputs ~ {'+'.join(features)}", df).fit()
    lr_time = time() - lr_start_time

    terms = [f"mul({model.params[feature]}, {feature})" for feature in features]
    lr_formula = reduce(
        lambda acc, term: f"add({acc}, {term})", terms, model.params["Intercept"]
    )

    gp = GP(
        df=df,
        features=features,
        outcome="outputs",
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

    gp_start_time = time()
    gp_lr_result = gp.run_gp(ngen=100, seeds=[lr_formula])
    gp_lr_end_time = time()
    gp_no_repair_seed_result = gp.run_gp(ngen=100, seeds=[lr_formula], repair=False)
    gp_no_repair_seed_end_time = time()
    gp_no_repair_no_seed_result = gp.run_gp(ngen=100, seeds=[], repair=False)
    gp_no_repair_no_seed_end_time = time()

    equation["simplified_formula"] = str(gp.simplify(equation["formula"]))

    equation["lr_raw_formula"] = str(lr_formula)
    equation["lr_simplified_formula"] = str(gp.simplify(lr_formula))
    equation["lr_nrmse"] = gp.fitness(lr_formula)[0]
    equation["lr_time"] = lr_time

    equation["gp_lr_raw_formula"] = str(gp_lr_result)
    equation["gp_lr_simplified_formula"] = str(gp.simplify(gp_lr_result))
    equation["gp_lr_nrmse"] = gp_lr_result.fitness.values[0]
    equation["gp_lr_time"] = gp_lr_end_time - gp_start_time

    equation["gp_no_repair_seed_raw_formula"] = str(gp_no_repair_seed_result)
    equation["gp_no_repair_seed_simplified_formula"] = str(gp.simplify(gp_no_repair_seed_result))
    equation["gp_no_repair_seed_nrmse"] = gp_no_repair_seed_result.fitness.values[0]
    equation["gp_no_repair_seed_time"] = gp_no_repair_seed_end_time - gp_lr_end_time

    equation["gp_no_repair_no_seed_raw_formula"] = str(gp_no_repair_no_seed_result)
    equation["gp_no_repair_no_seed_simplified_formula"] = str(gp.simplify(gp_no_repair_no_seed_result))
    equation["gp_no_repair_no_seed_nrmse"] = gp_no_repair_no_seed_result.fitness.values[0]
    equation["gp_no_repair_no_seed_end_time"] = gp_no_repair_no_seed_end_time - gp_no_repair_seed_end_time

    equation.pop("training_data")
    equation["outputs"] = df["outputs"].to_list()

if args.outfile:
    with open(args.outfile, 'w') as f:
        json.dump(equations, f)
