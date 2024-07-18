# RQ1: To what extent does the use of GP for linear regression create a more accurate computational model?
# RQ2: How does changing the operators used affect the accuracy of our computational model?
# RQ3: How does changing the size of the data set used affect the accuracy of our computational model?
# RQ4: How does the use of a noisy data set affect the accuracy of our computational model?

# Accuracy: How many generations for fitness = 0 or if gen > 40 use nmse - I need to figure out how to give a concrete score that combines these ideas

# Experiment:
# 1. Create a list of 10 equations of different complexity and using different operators.
# 2. Run 25 different seeds for each equation using my mix of GP + linear regression.
# 2.1 RQ2: 4 using just add, sub, mul, div. 4 adding neg, cos, sin + random. 5 using all.
# 2.2 RQ3: 7 using 10/25/50/100/250/600/1000) data points in range x (2-1002 randomly).
# 2.3 RQ4: 5 using noisty data.
# 3. Run those 25 seeds using just GP.
# 4. Run those 25 seeds using just linear regression

# Ideas:
# Syntatic closenose (RQ5?) - have to research
# If a regression coefficient close to 0 = remove - post processing

import random
import warnings
import patsy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels

from functools import partial
from deap import algorithms, base, creator, tools, gp

from numpy import negative, exp, power, log, sin, cos, tan, sinh, cosh, tanh

from operator import attrgetter, add, sub, mul, truediv

warnings.filterwarnings("error")

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(add, 2)
pset.addPrimitive(sub, 2)
pset.addPrimitive(mul, 2)
pset.addPrimitive(truediv, 2)
pset.addPrimitive(negative, 1)
pset.addPrimitive(cos, 1)
pset.addPrimitive(sin, 1)
pset.addEphemeralConstant("rand101", partial(random.randint, -1, 1))
pset.renameArguments(ARG0="x")

pset.addPrimitive(exp, 1)
pset.addPrimitive(log, 1)
pset.addPrimitive(power, 2)
pset.addPrimitive(tan, 1)
# pset.addPrimitive(sinh, 1)
# pset.addPrimitive(cosh, 1)
# pset.addPrimitive(tanh, 1)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalSymbReg(individual, points_x, points_y):
    points_x["y"] = points_y
    try:
        # Create model, fit (run) it, give estimates from it]
        model = smf.ols(f"y ~ {individual}", points_x)
        res = model.fit()
        y_estimates = res.predict(points_x)

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
    ):
        return (10**100,)


points_x = pd.DataFrame({"x": [float(x) for x in range(2, 100)]})
points_y = pd.Series([(log(x**5) + (x**3 + x**2 + x**0.5) / log(x**3.5) + exp(x)) for x in range(2, 100)])

toolbox.register("evaluate", evalSymbReg, points_x=points_x, points_y=points_y)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=lambda x: x.height + 1, max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=lambda x: x.height + 1, max_value=17))


def main():
    random.seed(1)

    pop = toolbox.population(n=300)
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

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, 10, 50, 0.4, 0.3, 20, stats=mstats, halloffame=hof, verbose=True)

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

    # print log
    return pop, log, hof


if __name__ == "__main__":
    pop, log, hof = main()
    print(pd.DataFrame({"x": [str(x) for x in pop], "fitness": [x.fitness for x in pop]}))
