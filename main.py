import numpy as np
import random
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt

from tree import Tree


def given_function(x, dim):
    if dim == 1:
        return 100*(-x[0]**2)**2 + (1 - x[0])**2
    else:
        return sum(100*(x[i + 1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))
    #return sum(x[i] ** (dim + 1) for i in range(len(x)))


def mse(function, dim, tree, points):
    with_function = [function(point, dim) for point in points]
    with_tree = [tree.compute(point) for point in points]

    return np.square(np.subtract(with_function, with_tree)).mean()


class GeneticProgram:
    def __init__(
            self,
            population_size,
            generation_max,
            crossover_prob,
            mutation_prob,
            max_height,
            points,
    ):
        self.population_size = population_size
        self.generation_max = generation_max
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        self.max_height = max_height
        self.points = points
        self.dim = len(points[0])
        self.population = self.generate_population()

    def generate_population(self):
        population = []
        for _ in range(self.population_size):
            tree = Tree.generate(self.max_height, self.dim)
            population.append(tree)
        return population

    def execute(self):
        for i in range(self.generation_max):
            self.mutate()
            self.crossover()
            self.reproduce(i)

    def mutate(self):
        for i in range(len(self.population)):
            if random.random() < self.mutation_prob:
                nodes = self.population[i].get_nodes()
                node, height = random.choice(list(nodes.items()))

                if height == 1:  # then it is root
                    tree = Tree.generate(self.max_height, self.dim)
                    # without true_divide(), pow() and exp() this is not needed:
                    # while not tree.is_good(self.points):
                    #     tree = Tree.generate(self.max_height, self.dim)
                    self.population[i] = tree
                    continue

                parent = node.parent
                no_of_child = parent.children.index(node)
                parent.children.pop(no_of_child)

                replacement = Tree.generate(self.max_height, self.dim, height).root
                parent.children.insert(no_of_child, replacement)
                replacement.parent = parent
    def crossover(self):
        def find_appropriate(a_nodes, b_nodes):
            for a_node, a_height in a_nodes:
                for b_node, b_height in b_nodes:
                    a_subtree_height = Tree.get_subtree_height(a_node)
                    b_subtree_height = Tree.get_subtree_height(b_node)

                    if ((a_height - 1 + b_subtree_height) <= self.max_height
                            and (b_height - 1 + a_subtree_height) <= self.max_height):
                        return a_node, b_node

            return None, None

        for i in range(self.population_size // 2):
            if random.random() < self.crossover_prob:
                a_tree = deepcopy(self.population[2 * i])
                b_tree = deepcopy(self.population[2 * i + 1])

                a_nodes = list(a_tree.get_nodes().items())[1:]  # removing root
                b_nodes = list(b_tree.get_nodes().items())[1:]
                random.shuffle(a_nodes)
                random.shuffle(b_nodes)

                a_node, b_node = find_appropriate(a_nodes, b_nodes)

                if a_node is None or b_node is None:
                    continue

                a_parent = a_node.parent
                a_no_of_child = a_parent.children.index(a_node)
                a_parent.children.pop(a_no_of_child)

                b_parent = b_node.parent
                b_no_of_child = b_parent.children.index(b_node)
                b_parent.children.pop(b_no_of_child)

                b_parent.children.insert(b_no_of_child, a_node)
                a_node.parent = b_parent

                a_parent.children.insert(a_no_of_child, b_node)
                b_node.parent = a_parent

                self.population[2 * i] = a_tree
                self.population[2 * i + 1] = b_tree

    def draw_pic(self, tree, current_generation):
        plt.clf()
        y = self.points
        x_tree = [tree.compute(point) for point in self.points]
        x_func = [given_function(point, self.dim) for point in self.points]

        plt.plot(y, np.c_[x_tree, x_func], label=['x_tree', 'x_func'])
        plt.legend()
        plt.savefig(f"results/{current_generation}.jpg")
    def reproduce(self, current_generation):
        values = np.array(
            list(
                map(lambda tree: -self.target_func(tree, self.points), self.population)
            )
        )
        hihi = min(self.population, key = lambda r: self.target_func(r, self.points))
        calc_by_tree = [f"{hihi.compute(point):.2f}" for point in self.points]
        calc_by_func = [f"{given_function(point, self.dim):.2f}" for point in self.points]
        print(f"Current generation: {current_generation}\n"
              f"Computed tree: ")
        hihi.print()
        print(
              f"Result(tree): {calc_by_tree}\n vs. result(func): {calc_by_func}\n"
              f"MSE: {-min(values)}."
              )
        self.draw_pic(hihi, current_generation)

        values -= min(values)
        values += 1

        sum = np.sum(values)
        values /= sum

        self.population = random.choices(
            self.population, values, k=self.population_size
        )
    def target_func(self, tree, points):
        return mse(given_function, self.dim, tree, points)

if __name__ == "__main__":
    points = [[x] for x in np.arange(-2.048, 2.048, 0.1)]
    print(points)
    GeneticProgram(100, 500, 0.7, 0.01, 10, points).execute()