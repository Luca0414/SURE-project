# Original
# RQ: To what extend does the use of GP for linear regression create a more accurate computational model?
# Accuracy: Like original fitness func - how close to the 'real' model (We need to give one)
# To ensure robustness test on: Different operators, Different data sizes, Noisy data

# Ideas:
# Make last bit diff research questions
# If found fitness = 0 stop - how many gens to find (RQ) - how long does it take
# Syntatic closenose (RQ) - have to research

import operator
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from functools import partial
from deap import algorithms, base, creator, tools, gp

# Define new functions

# Fitness value of infinite if error - not return 1
def protectedDiv(left, right):
    try:
        return left / right
    except:
        return 1

def protectedLog(value):
    try:
        return math.log(value)
    except:
        return 1

def protectedPow(base, exponent):
    try:
        return math.pow(base, exponent)
    except:
        return 1
    
def protectedSinh(x):
    try:
        return math.sinh(x)
    except:
        return 1

def protectedCosh(x):
    try:
        return math.cosh(x)
    except:
        return 1
    
def protectedExp(x):
    try:
        return math.exp(x)
    except:
        return 1

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", partial(random.randint, -1, 1))
pset.renameArguments(ARG0='x')

pset.addPrimitive(protectedExp, 1)
pset.addPrimitive(protectedLog, 1)
pset.addPrimitive(protectedPow, 2)
pset.addPrimitive(math.tan, 1)
pset.addPrimitive(protectedSinh, 1)
pset.addPrimitive(protectedCosh, 1)
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

def evalSymbReg(individual, points_x, points_y): # Convert to pandas instead of numpy.array

    terms = split(individual)
    X = np.empty((0, len(terms) + 1))

    # Create Exog variables
    for x in points_x:
        values = []
        for term in terms:
            func  = toolbox.compile(expr=term)
            values.append(func(x))
        values.append(1) #add intercepts
        X = np.vstack([X, values])

    
    #Idea: If a regression coefficient close to 0 - remove - post processing
    

    # Create model, fit (run) it, give estimates from it
    model = sm.OLS(points_y, X)
    res = model.fit()
    y_estimates = res.predict(X)

    # print(individual)

    # Calc errors
    sqerrors = (y_estimates - points_y)**2

    # improve with michael's function - https://www.marinedatascience.co/blog/2019/01/07/normalizing-the-rmse/
    return math.fsum(sqerrors) / len(points_x),

points_x = [x for x in range(-100, 100)]
points_y = [(protectedLog(x**4) + protectedDiv((x**3 + x**2 + x**0.5), protectedLog(x**5)) + math.exp(x)) for x in range(-100, 100)]

toolbox.register("evaluate", evalSymbReg, points_x=points_x, points_y=points_y)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    random.seed(2)

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