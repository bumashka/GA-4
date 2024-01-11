import random
import sys
import time
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt

from tree import Tree


def given_function(x, dim):
    if dim == 1:
        return 100 * (-x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    else:
        return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


def mse(function, dim, tree, points):
    with_function = [function(point, dim) for point in points]
    with_tree = [tree.compute(point) for point in points]

    return np.square(np.subtract(with_function, with_tree)).mean()


class FGA:

    def __init__(self, config):
        # Максимальное количество хромосом в популяции
        self.max_population = config['max_population']
        # Максимальное количество эпох
        self.max_epochs = config['max_epochs']
        self.points = [[x] for x in np.arange(-2.048, 2.048, 0.1)]
        self.dim = len(self.points[0])
        self.max_height = config['max_height']
        # Актуальная эпоха
        self.current_epoch = 0
        # Вероятность кроссовера
        self.crossover_chance = config['crossover_chance']
        # Вероятность мутации
        self.mutation_chance = config['mutation_chance']
        # Актуальная популяция хромосом
        # Потомки актуальной популяции
        self.children = []
        # Лучшее решение актуальной популяции
        self.current_best_solution = sys.maxsize - 1
        self.population = self.generate_population()

    def target_func(self, tree, points):
        return mse(given_function, self.dim, tree, points)

    def generate_population(self):
        population = []
        for _ in range(self.max_population):
            tree = Tree.generate(self.max_height, self.dim)
            population.append(tree)
        return population

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

        for i in range(self.max_population // 2):
            if random.random() < self.crossover_chance:
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

    def mutate(self):
        for i in range(len(self.population)):
            if random.random() < self.mutation_chance:
                nodes = self.population[i].get_nodes()
                node, height = random.choice(list(nodes.items()))

                if height == 1:  # then it is root
                    tree = Tree.generate(self.max_height, self.dim)
                    self.population[i] = tree
                    continue

                parent = node.parent
                no_of_child = parent.children.index(node)
                parent.children.pop(no_of_child)

                replacement = Tree.generate(self.max_height, self.dim, height).root
                parent.children.insert(no_of_child, replacement)
                replacement.parent = parent

    def reproduce(self, current_generation):
        values = np.array(
            list(
                map(lambda tree: -self.target_func(tree, self.points), self.population)
            )
        )
        hihi = min(self.population, key=lambda r: self.target_func(r, self.points))
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
            self.population, values, k=self.max_population
        )

    def run(self):
        for i in range(self.max_epochs):
            self.mutate()
            self.crossover()
            self.reproduce(i)


def main():
    for cross_chance in [0.25, 0.5, 0.7, 0.9]:
        for mut_chance in [0.001, 0.01, 0.1, 0.5, 0.8]:
            for max_population in [50, 100, 150]:
                config = {
                    "crossover_chance": cross_chance,
                    "mutation_chance": mut_chance,
                    "max_population": max_population,
                    "max_epochs": 100,
                    "max_height": 10
                }
                GA = FGA(config)
                GA.run()


if __name__ == '__main__':
    random.seed(round(time.time()))
    main()
