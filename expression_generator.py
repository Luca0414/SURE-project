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


class ExpressionGenerator:
    def __init__(self, n_var, random_seed=0):
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.n_var = n_var

        # Setup the primitive set
        self.pset = gp.PrimitiveSetTyped("main", [float] * n_var, float)
        self.pset.renameArguments(**{f"ARG{i}": f"x{i}" for i in range(n_var)})
        self.pset.addPrimitive(add, [float, float], float)
        self.pset.addPrimitive(mul, [float, float], float)
        self.pset.addPrimitive(np.sin, [float], float)
        self.pset.addPrimitive(np.cos, [float], float)
        self.pset.addPrimitive(np.tan, [float], float)
        self.pset.addPrimitive(np.log, [float], float)
        self.pset.addPrimitive(reciprocal, [float], float)
        self.pset.addPrimitive(root, [float], float)
        self.pset.addPrimitive(power_2, [float], float)
        self.pset.addPrimitive(power_3, [float], float)

        creator.create("Individual", gp.PrimitiveTree)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genFull, pset=self.pset, min_=0, max_=4)
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.expr
        )
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def make_term(self, var):
        term = self.toolbox.individual()
        _, _, labels = gp.graph(term)
        if var in labels.values():
            return term
        terminals = [
            k
            for k, v in labels.items()
            if isinstance(self.pset.mapping[v], gp.Terminal)
        ]
        index = 0 if len(terminals) == 0 else random.choice(terminals)
        term[index] = self.pset.mapping[var]
        return term

    def add_noise(self, array, noise_factor):
        noise = np.zeros(len(array))
        noise = np.random.uniform(-1, 1, size=array.size) * noise_factor * array
        return array + noise

    def generate_training_data(self, individual, size: int):
        individual = creator.Individual(
            gp.PrimitiveTree.from_string(individual, self.pset)
        )
        func = self.toolbox.compile(expr=individual)
        data = pd.DataFrame(
            np.random.uniform(0, 100, size=(size, self.n_var)),
            columns=[f"x{n}" for n in range(self.n_var)],
        )
        outputs = np.array([func(**row.to_dict()) for _, row in data.iterrows()])

        return data, outputs

    def generate_expression(self):
        terms = [str(self.make_term(arg)) for arg in self.pset.arguments]
        terms = [f"mul({np.random.uniform(-10, 10)}, {term})" for term in terms]
        individual = reduce(
            lambda acc, term: f"add({acc}, {term})", terms, random.uniform(-10, 10)
        )
        return individual


if __name__ == "__main__":
    index = 0
    np.random.seed(1)
    random.seed(1)

    experimental_runs = []
    for n_var in range(1, 11):
        generator = ExpressionGenerator(n_var)
        # Generate 30 expressions for each number of variables
        for _ in range(30):
            is_null = True
            while is_null:
                individual = generator.generate_expression()
                data, outputs = generator.generate_training_data(individual)

                # If there are any null values or if we get a trivially constant output, try again
                is_null = data.isnull().any().any() or len(set(outputs)) == 1
            experimental_runs.append(
                {
                    "index": index,
                    "formula": str(individual),
                    "num_vars": n_var,
                    "training_data": data[[f"x{n}" for n in range(n_var)]].to_dict(),
                    "outputs": outputs,
                }
            )
            index += 1
    with open("equations.json", "w") as f:
        json.dump(experimental_runs, f)
