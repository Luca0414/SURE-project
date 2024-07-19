# RQ1: To what extent does the use of GP for linear regression create a more accurate computational model than pure GP or linear regression?
# RQ2: How does changing the operators used affect the accuracy of our computational model?
# RQ3: How does changing the size of the data set used affect the accuracy of our computational model?
# RQ4: How does the use of a noisy data set affect the accuracy of our computational model?

# Accuracy: How many generations for fitness = 0 or if gen > 40 use min nmse - Will have to use two figures to show these two outcomes

# Experiment: 
# 1. Create a list of 10(00 in real) equations of different complexity and using different operators. # Use benchmarks given
# 2. Run 30 different seeds for each version.
# 2.1 RQ2: Using just add, sub, mul, div, neg + reciprocal. Adding sin, cos, tan, log, root + square. Adding exponential, cube, 4th power, sinh, cosh + tanh. 
# 2.2 RQ3: Using 10/100/1000 data points randomly for each variable. 
# 2.3 RQ4: Using noisty data.
# 3. Run those 150 seeds using just GP.
# 4. Run those 150 seeds using just linear regression.

# Ideas:
# Syntatic closenose (RQ5?) - have to research
# If a regression coefficient close to 0 = remove - post processing
# Re-add 25/50/250/500 data sizes?
# Justify use of operators + data numbers
# Look into more than one variable input (Michael looking into it)
# save to json instead of default when using bash

import random
import warnings
import patsy
import argparse
import statsmodels

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from deap import base, creator, tools, gp
from numpy import negative, exp, log, sin, cos, tan, sinh, cosh, tanh
from operator import add, sub, mul, truediv

random.seed(1) # change to use argument


warnings.filterwarnings("error")

# parser = argparse.ArgumentParser(description="Run experiments")
# parser.add_argument('--seed', type=int, required=True, help='Random seed')
# parser.add_argument('--equation', type=str, required=True, help='Equation to use')
# parser.add_argument('--operators', type=str, required=True, help='Operators to include')
# parser.add_argument('--data_points', type=int, required=True, help='Number of data points')
# parser.add_argument('--noise', type=float, required=True, help='Noise level')
# parser.add_argument('--method', type=str, required=True, choices=['GP', 'LR', 'GPLR'], help='Method to use (GP, LR)')

# args = parser.parse_args()

# Define more operators
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

pset = gp.PrimitiveSet("MAIN", 1)
pset.renameArguments(ARG0='x')

pset.addPrimitive(add, 2)
pset.addPrimitive(sub, 2)
pset.addPrimitive(mul, 2)
pset.addPrimitive(truediv, 2)
pset.addPrimitive(negative, 1)
pset.addPrimitive(reciprocal, 1)

pset.addPrimitive(sin, 1)
pset.addPrimitive(cos, 1)
pset.addPrimitive(tan, 1)
pset.addPrimitive(log, 1)
pset.addPrimitive(root, 1)
pset.addPrimitive(square, 1)

pset.addPrimitive(exp, 1)
pset.addPrimitive(cube, 1)
pset.addPrimitive(fourth_power, 1)
pset.addPrimitive(sinh, 1)
pset.addPrimitive(cosh, 1)
pset.addPrimitive(tanh, 1)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

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
        y_estimates = res.predict(df)

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
    ) as e:
        return individual

def evalSymbReg(individual, points_x, points_y):
    try:
        # Create model, fit (run) it, give estimates from it]
        func = gp.compile(individual, pset)
        y_estimates = pd.Series([func(**x) for _, x in points_x.iterrows()])

        # Calc errors using an improved normalised mean squared
        sqerrors = (points_y - y_estimates) ** 2
        mean_squared = sqerrors.sum() / len(points_x)
        nmse = mean_squared / (points_y.sum() / len(points_y))

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


points_x = pd.DataFrame({"x": [float(x) for x in range(2, 100)]})
points_y = pd.Series([x**2 + x + 5 for x in range(2, 100)])

solution = gp.PrimitiveTree.from_string("add(square(x), x)", pset)
print(repair(solution, points_x, points_y))
print(solution, evalSymbReg(solution, points_x, points_y))


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


toolbox.register("evaluate", evalSymbReg, points_x=points_x, points_y=points_y)
toolbox.register("repair", repair, points_x=points_x, points_y=points_y)
toolbox.register("select", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", mutation, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=lambda x: x.height + 1, max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=lambda x: x.height + 1, max_value=17))


def main():

    mu = 20

    pop = toolbox.population(n=mu)

    hof = tools.HallOfFame(
        1
    )  # to maintain some individuals if using ea/ga that deletes old population on each generation

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
        lambda_=5,
        ngen=100,
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

    plt.show()

    # print log
    return pop, log, hof


if __name__ == "__main__":
    pop, log, hof = main()
    print(len(pop))
    print(
        pd.DataFrame({"x": [str(x) for x in pop], "fitness": [x.fitness for x in pop]})
    )