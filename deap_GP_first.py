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

import operator
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from functools import partial
from deap import algorithms, base, creator, tools, gp

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.truediv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", partial(random.randint, -1, 1))
pset.renameArguments(ARG0='x')

pset.addPrimitive(math.exp, 1)
pset.addPrimitive(math.log, 1)
pset.addPrimitive(math.pow, 2)
pset.addPrimitive(math.tan, 1)
pset.addPrimitive(math.sinh, 1)
pset.addPrimitive(math.cosh, 1)
pset.addPrimitive(math.tanh, 1)

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

def evalSymbReg(individual, points_x, points_y):
    try:
        terms = split(individual)

        # Create Exog variables as a pandas DataFrame
        data = []
        for x in points_x:
            values = []
            for term in terms:
                func  = toolbox.compile(expr=term)
                values.append(func(x))
            values.append(1) #add intercepts
            data.append(values)
        
        df = pd.DataFrame(data)        

        # Create model, fit (run) it, give estimates from it
        model = sm.OLS(points_y, df)
        res = model.fit()
        y_estimates = res.predict(df)

        # Calc errors using an improved normalised mean squared
        sqerrors = (y_estimates - points_y)**2
        mean_squared = math.fsum(sqerrors) / len(points_x)
        nmse = mean_squared / (math.fsum(points_y)/ len(points_y))

        return nmse,

        # Fitness value of infinite if error - not return 1
    except (OverflowError, ValueError, ZeroDivisionError):
        return 10**100,

points_x = [x for x in range(2, 100)]
points_y = [(math.log(x**5) + (x**3 + x**2 + x**0.5)/math.log(x**3.5) + math.exp(x)) for x in range(2, 100)]

toolbox.register("evaluate", evalSymbReg, points_x=points_x, points_y=points_y)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    random.seed(1)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1) # to maintain some individuals if using ea/ga that deletes old population on each generation

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, 10, 50, 0.4, 0.3, 40, stats=mstats,
                                   halloffame=hof, verbose=True)

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
    main()