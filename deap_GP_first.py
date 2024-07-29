# RQ1: To what extent does the use of GP for linear regression create a more accurate computational model than pure GP or linear regression?
# RQ2: How does changing the operators used affect the accuracy of our computational model?
# RQ3: How does changing the size of the data set used affect the accuracy of our computational model?
# RQ4: How does the use of a noisy data set affect the accuracy of our computational model?

# Accuracy: How many generations for fitness = 0 or if gen > 40 use min nmse - Will have to use two figures to show these two outcomes

# Experiment: 
# 1. Make 100 equations for each amount of variables 1-10. And the benchmarks given: All but salustowicz_2d + unwrapped_ball
# 2. Run 30 different seeds for each version.
# 2.1 RQ2: one using all operators, one removing 1 operator thats in equation and one removing 2 operator that is in equation
# 2.2 RQ3: Using 10/100/1000 data points randomly for each variable.
# 2.3 RQ4: Using noisty data.
# 3. Run those 150 seeds using just GP.
# 4. Run those 150 seeds using just linear regression.

# Ideas:
# Syntatic closenose (RQ5?) - have to research
# If a regression coefficient close to 0 = remove - post processing
# Re-add 25/50/250/500 data sizes?

import random
import warnings
import patsy
import argparse
import statsmodels
import time
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from deap import base, creator, tools, gp
from numpy import negative, log, sin, cos, tan, exp
from operator import add, sub, mul, truediv

def root(x):
    return x ** 0.5

def square(x):
    return x ** 2

def cube(x):
    return x ** 3

def fourth_power(x):
    return x ** 4

def reciprocal(x):
    return 1 / x

def make_primitive_set():
    pset = gp.PrimitiveSet("START", args.num_vars)

    pset.addPrimitive(add, 2)
    pset.addPrimitive(sub, 2)
    pset.addPrimitive(mul, 2)
    pset.addPrimitive(truediv, 2)
    pset.addPrimitive(negative, 1)
    pset.addPrimitive(sin, 1)
    pset.addPrimitive(cos, 1)
    pset.addPrimitive(tan, 1)
    pset.addPrimitive(log, 1)
    pset.addPrimitive(reciprocal, 1)
    pset.addPrimitive(root, 1)
    pset.addPrimitive(square, 1)
    pset.addPrimitive(cube, 1)
    pset.addPrimitive(fourth_power, 1)

    equation = gp.PrimitiveTree.from_string(args.equation, pset)

    primitive_functions = [add, sub, mul, truediv, negative, sin, cos, tan, log, reciprocal, root, square, cube, fourth_power]
    primitive_aritys = [2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    primitive_names = ['add', 'sub', 'mul', 'truediv', 'negative', 'sin', 'cos', 'tan', 'log', 'reciprocal', 'root', 'square', 'cube', 'fourth_power']

    primitives_used = []
    for node in equation: # Change to use equation
        if node not in primitives_used and isinstance(node, gp.Primitive) and node.name not in ['sub','add','mul']: # mul because of when using LR
            primitives_used.append(node)

    if args.operator_level > 0:
        random_prim_used_1 = primitives_used[random.randint(0, len(primitives_used) - 1)]
        remove_prim_pos_1 = primitive_names.index(random_prim_used_1.name) 

        primitive_functions.pop(remove_prim_pos_1)
        primitive_names.pop(remove_prim_pos_1)
        primitive_aritys.pop(remove_prim_pos_1)

        primitives_used.remove(random_prim_used_1)
    if args.operator_level > 1:
        random_prim_used_2 = primitives_used[random.randint(0, len(primitives_used) - 1)]
        remove_prim_pos_2 = primitive_names.index(random_prim_used_2.name) 

        primitive_functions.pop(remove_prim_pos_2)
        primitive_names.pop(remove_prim_pos_2)
        primitive_aritys.pop(remove_prim_pos_2)

        primitives_used.remove(random_prim_used_2)
    if args.operator_level > 2:
        random_prim_used_3 = primitives_used[random.randint(0, len(primitives_used) - 1)]
        remove_prim_pos_3 = primitive_names.index(random_prim_used_3.name) 

        primitive_functions.pop(remove_prim_pos_3)
        primitive_names.pop(remove_prim_pos_3)
        primitive_aritys.pop(remove_prim_pos_3)

        primitives_used.remove(random_prim_used_3)
    
    pset_final = gp.PrimitiveSet("MAIN", args.num_vars)
    for index, func in enumerate(primitive_functions):
        pset_final.addPrimitive(func, primitive_aritys[index])

    return pset_final, gp.compile(equation, pset)

def split(individual):
    if len(individual) > 1:
        terms = []
        # Recurse over children if add/sub
        if individual[0].name in ['add','sub']:
            terms.extend(split(creator.Individual(gp.PrimitiveTree(individual[individual.searchSubtree(1).start:individual.searchSubtree(1).stop]))))
            terms.extend(split(creator.Individual(gp.PrimitiveTree(individual[individual.searchSubtree(1).stop:]))))
        else:
            terms.append(individual)
        return terms
    return [individual]

def repair(individual, points_x, points_y):
    eq = f"y ~ {' + '.join(str(x) for x in split(individual))}"
    df = pd.concat((points_x, points_y.rename("y")), axis=1)
    try:
        # Create model, fit (run) it, give estimates from it]
        model = smf.ols(eq, df)
        res = model.fit()

        eqn = f"{res.params['Intercept']}"
        for term, coefficient in res.params.items():
            if term != "Intercept":
                eqn = f"add({eqn}, mul({coefficient}, {term}))"
        repaired = type(individual)(gp.PrimitiveTree.from_string(eqn, pset))
        return repaired
    except (
        OverflowError,
        ValueError,
        ZeroDivisionError,
        statsmodels.tools.sm_exceptions.MissingDataError,
        patsy.PatsyError,
        np.core._exceptions._UFuncOutputCastingError,
        np.linalg.LinAlgError
    ) as e:
        # print(e)
        return individual

def evalSymbReg(individual, points_x, points_y):
    try:
        # Create model, fit (run) it, give estimates from it]
        func = gp.compile(individual, pset)
        y_estimates = pd.Series([func(**x) for _, x in points_x.iterrows()])
        for estimate in y_estimates:
            if np.iscomplex(estimate):
                raise ValueError("Estimate is Complex")

        # Calc errors using an improved normalised mean squared
        sqerrors = (points_y - y_estimates) ** 2
        mean_squared = sqerrors.sum() / len(points_x)
        nmse = mean_squared / len(points_y)

        return (nmse,)

        # Fitness value of infinite if error - not return 1
    except (
        OverflowError,
        ValueError,
        ZeroDivisionError,
        statsmodels.tools.sm_exceptions.MissingDataError,
        patsy.PatsyError,
        RuntimeWarning,
    ) as e:
        print(e)
        return (float("inf"),)

def make_offspring(population, toolbox, lambda_):
    offspring = []
    for i in range(lambda_):
        parent1, parent2 = tools.selTournament(population, 2, 2)
        child, _ = toolbox.mate(toolbox.clone(parent1), toolbox.clone(parent2))
        del child.fitness.values
        (child,) = toolbox.mutate(child)
        offspring.append(child)
    return offspring

def mutation(individual, expr, pset):
    choice = random.randint(0, 2)
    if choice == 0:
        mutated = gp.mutUniform(toolbox.clone(individual), expr, pset)
    elif choice == 1:
        mutated = gp.mutNodeReplacement(toolbox.clone(individual), pset)
    # elif choice == 2:
    #     mutated = gp.mutInsert(toolbox.clone(individual), pset)
    elif choice == 2:
        mutated = gp.mutShrink(toolbox.clone(individual))
    else:
        raise ValueError("Invalid mutation choice")
    return mutated


def eaMuPlusLambda(
    population,
    toolbox,
    mu,
    lambda_,
    ngen,
    stats=None,
    halloffame=None,
    verbose=__debug__,
):
    if args.method == 'GPLR':
        population = [toolbox.repair(ind) for ind in population]

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Vary the population
        offspring = make_offspring(population, toolbox, lambda_)
        if args.method == 'GPLR':
            offspring = [toolbox.repair(ind) for ind in offspring]

        # Evaluate the individuals with an invalid fitness
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(offspring), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def main():
    mu = 100

    pop = toolbox.population(n=mu)
    if args.method == 'GP' or args.method == 'GPLR':
        hof = tools.HallOfFame(1)  # to maintain some individuals if using ea/ga that deletes old population on each generation

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        pop, log = eaMuPlusLambda(
            pop,
            toolbox,
            mu=mu,
            lambda_=50,
            ngen=125,
            stats=None,
            halloffame=hof,
            verbose=True,
        )

        gen = log.chapters["fitness"].select("gen")
        min = log.chapters["fitness"].select("min")
        avg = log.chapters["fitness"].select("avg")

        fig, ax1 = plt.subplots()
        line1 = ax1.plot(gen, min, "b", label="Minimum")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("fitness", color="b")
        for tl in ax1.get_yticklabels():
            tl.set_color("b")

        line2 = ax1.plot(gen, avg, "r-", label="Average")

        lns = line1 + line2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="center right")

        # plt.show()

        return pop
    else:
        pop = [toolbox.repair(ind) for ind in pop]
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)

        return pop


if __name__ == "__main__":
    start_time = time.time()

    warnings.filterwarnings("error")

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--equation', type=str, required=True)
    parser.add_argument('--operator_level', type=int, required=True)
    parser.add_argument('--data_points', type=int, required=True)
    parser.add_argument('--noise', type=float, required=True)
    parser.add_argument('--method', type=str, required=True, choices=['GP', 'LR', 'GPLR'])
    parser.add_argument('--num_vars', type=int, required=True)

    args = parser.parse_args()

    random.seed(args.seed)

    pset, equation = make_primitive_set()

    data = {}
    valid_results = []
    num_deletes = 0

    for i in range(args.num_vars):
        column_name = f"ARG{i}"
        data[column_name] = [random.randint(2, 100) for _ in range(args.data_points)]

    points_x_temp = pd.DataFrame(data)

    for row in range(args.data_points):
        try:
            y_value = equation(*points_x_temp.iloc[row])
            valid_results.append(y_value + random.gauss(0, (args.noise * y_value)))
        except Exception as e:
            print(e)
            print(row)
            num_deletes += 1
            for i in range(args.num_vars):
                column_name = f"ARG{i}"
                data[column_name].pop(row-num_deletes)
    
    points_x = pd.DataFrame(data)
    points_y = pd.Series(valid_results)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evalSymbReg, points_x=points_x, points_y=points_y)
    toolbox.register("repair", repair, points_x=points_x, points_y=points_y)
    toolbox.register("select", tools.selBest)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", mutation, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=lambda x: x.height + 1, max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=lambda x: x.height + 1, max_value=17))

    pop = main()

    output = {
        "equation": args.equation,
        "num_vars": args.num_vars,
        "pset": [prim.name for prim in list(pset.primitives.values())[0]],
        "num_data_points": args.data_points,
        "noise": args.noise,
        "method": args.method,
        "runtime": time.time() - start_time,
        "population": [str(x) for x in pop],
        "fitnesses": [x.fitness.values[0] for x in pop if x.fitness.values[0] != float("inf")]
    }

    filename = 'results.json'

    try:
        with open(filename, 'r') as file:
            existing_data = json.load(file)
    except:
        existing_data = []

    existing_data.append(output)

    with open(filename, 'w') as file:
        json.dump(existing_data, file, indent=4)

    print(pd.DataFrame({"x": [str(x) for x in pop], "fitness": [x.fitness for x in pop]}))