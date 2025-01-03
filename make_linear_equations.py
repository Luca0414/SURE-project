from deap import base, creator, tools, gp
from operator import add, sub, mul, truediv
from functools import reduce
import pandas as pd
import numpy as np
import random
import json


def root(x):
    return x**0.5


def power_2(x):
    return x**2


def power_3(x):
    return x**3


def reciprocal(x):
    return 1 / x


def setup_pset(n_var):
    pset = gp.PrimitiveSetTyped("main", [float] * n_var, float)
    pset.renameArguments(**{f"ARG{i}": f"x{i}" for i in range(n_var)})
    pset.addPrimitive(add, [float, float], float)
    pset.addPrimitive(mul, [float, float], float)
    pset.addPrimitive(np.sin, [float], float)
    pset.addPrimitive(np.cos, [float], float)
    pset.addPrimitive(np.tan, [float], float)
    pset.addPrimitive(np.log, [float], float)
    pset.addPrimitive(reciprocal, [float], float)
    pset.addPrimitive(root, [float], float)
    pset.addPrimitive(power_2, [float], float)
    pset.addPrimitive(power_3, [float], float)
    return pset


def make_term(var, toolbox, pset):
    term = toolbox.individual()
    _, _, labels = gp.graph(term)
    if var in labels.values():
        return term
    terminals = [
        k for k, v in labels.items() if isinstance(pset.mapping[v], gp.Terminal)
    ]
    index = 0 if len(terminals) == 0 else random.choice(terminals)
    term[index] = pset.mapping[var]
    return term


def generate_individual(toolbox, pset):
    terms = [str(make_term(arg, toolbox, pset)) for arg in pset.arguments]
    terms = [f"mul({np.random.uniform(-10, 10)}, {term})" for term in terms]
    individual = reduce(
        lambda acc, term: f"add({acc}, {term})", terms, random.uniform(-10, 10)
    )
    individual = creator.Individual(gp.PrimitiveTree.from_string(individual, pset))
    return individual

if __name__ == "__main__":
    index = 0
    np.random.seed(1)
    random.seed(1)
    creator.create("Individual", gp.PrimitiveTree)

    experimental_runs = []
    for n_var in range(1, 11):
        pset = setup_pset(n_var)
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genFull, pset=pset, min_=0, max_=4)
        toolbox.register(
            "individual", tools.initIterate, creator.Individual, toolbox.expr
        )
        toolbox.register("compile", gp.compile, pset=pset)

        # Generate 30 expressions for each number of variables
        for _ in range(30):
            is_null = True
            while is_null:
                individual = generate_individual(toolbox, pset)
                func = toolbox.compile(expr=individual)
                data = pd.DataFrame(
                    np.random.uniform(0, 100, size=(100, n_var)),
                    columns=[f"x{n}" for n in range(n_var)],
                )
                features = list(data.columns)
                outputs = [func(**row.to_dict()) for _, row in data.iterrows()]
                data["predicted"] = outputs
                # If there are any null values or if we get a trivially constant output, try again
                is_null = data.isnull().any().any() or len(set(outputs)) == 1
            experimental_runs.append(
                {
                    "index": index,
                    "formula": str(individual),
                    "num_vars": n_var,
                    "training_data": data[features].to_dict(),
                    "outputs": outputs,
                }
            )
            index += 1
    with open("equations.json", 'w') as f:
        json.dump(experimental_runs, f)
