import random

from deap import base, creator, tools, gp
from numpy import negative, exp, log, sin, cos, tan
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
    
creator.create("Individual", gp.PrimitiveTree)
random.seed(1)

for n_var in range(1,11):
    pset = gp.PrimitiveSet("MAIN", n_var)

    pset.addPrimitive(add, 2)
    pset.addPrimitive(sub, 2)
    pset.addPrimitive(mul, 2)
    pset.addPrimitive(truediv, 2)
    pset.addPrimitive(negative, 1)
    pset.addPrimitive(sin, 1)
    pset.addPrimitive(cos, 1)
    pset.addPrimitive(tan, 1)
    pset.addPrimitive(log, 1)
    # pset.addPrimitive(exp, 1)
    pset.addPrimitive(reciprocal, 1)
    pset.addPrimitive(root, 1)
    pset.addPrimitive(square, 1)
    pset.addPrimitive(cube, 1)
    pset.addPrimitive(fourth_power, 1)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genFull, pset=pset, min_=6, max_=9)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pop = toolbox.population(n=1)
    for ind in pop:
        with open("small_equations.txt", "a") as f:
            f.write(str(ind) + "\n" + str(n_var) + "\n")

    # Create another GP that makes sure all Args are added properly
    # It adds test data to check for erros maybe?

