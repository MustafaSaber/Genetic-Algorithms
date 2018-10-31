import bisect
import math
import multiprocessing
import random
import time

import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import trange, tqdm

import FitnessFunction as ff
from graph import Graph
from Object import Object
from testcase import TestCase


class GeneticAlgorithm():
    def __init__(self, pop_size, generations, p_crossover, p_mutation, dependency_factor):
        self.pop_size = pop_size

        # Mumber of generations
        self.generations = generations

        # Probability of crossover between [ 0.4 , 0.7 ]
        self.p_crossover = p_crossover

        # Probability of mutation between [ 0.001 , 0.1 ]
        self.p_mutation = p_mutation
        self.dependency_factor = dependency_factor

    def create_population(self, degree):
        return [[random.uniform(-10, 10) for _ in range(degree + 1)] for _ in range(self.pop_size)]

    def mutate(self, chromosome, generation_number):
        for i in range(0, len(chromosome)):
            r1, r2, r3 = random.uniform(0, 1), random.uniform(
                0, 1), random.uniform(0, 1)
            val = -10 if r1 <= 0.5 else 10
            power = (1 - generation_number /
                     self.generations) ** self.dependency_factor
            delta = val * (1 - r2 ** power)
            if r3 <= self.p_mutation * (1 - generation_number / self.generations):
                chromosome[i] = random.uniform(-10, 10)

    def cross_over(self, parent1, parent2):
        r1 = random.randint(1, len(parent1) - 1)
        r2 = random.uniform(0, 1)
        parent1, parent2 = list(parent1), list(parent2)
        if r2 <= self.p_crossover:
            child1 = parent1[:r1] + parent2[r1:]
            child2 = parent2[:r1] + parent1[r1:]
            return child1, child2
        else:
            return parent1, parent2

    def cross_over_non_uniform(self, parent1, parent2):
        child = [list(parent1), list(parent2)]
        for i in range(len(parent1)):
            r2 = random.uniform(0, 1)
            if r2 <= self.p_crossover:
                # Swap
                child[0][i], child[1][i] = child[1][i], child[0][i]
        return child[0], child[1]

    @staticmethod
    def cumulative_sum(fitness_list):
        new = []
        total_sum = 0
        for i in fitness_list:
            total_sum += i
            new.append(total_sum)
        return new

    @staticmethod
    def pop_fitness(pop, points):
        return [ff.fitness(i, points) for i in pop]

    def genetic_algorithm(self, pop, points, generation_number):
        fitness_array = self.pop_fitness(pop, points)
        fitness_array_cumlative = self.cumulative_sum(fitness_array)
        summation = fitness_array_cumlative[len(fitness_array_cumlative) - 1]
        # Proof that the value converges
        # print("%f && %f " % (min(fitness_array), max(fitness_array)))
        next_pop = []
        while len(next_pop) < len(pop):
            r1, r2 = random.uniform(0, summation), random.uniform(0, summation)
            idx1 = bisect.bisect_left(fitness_array_cumlative, r1)
            idx2 = bisect.bisect_left(fitness_array_cumlative, r2)

            Offspring1, Offspring2 = self.cross_over_non_uniform(
                pop[idx1], pop[idx2])

            self.mutate(Offspring1, generation_number)
            self.mutate(Offspring2, generation_number)
            next_pop.append(Offspring1)
            next_pop.append(Offspring2)

        next_gen_fitness = self.pop_fitness(next_pop, points)
        pip_size = len(pop)
        # Take best of previous and best of new.
        next_gen = []
        for _ in range(len(pop) // 2):
            m1 = fitness_array.index(min(fitness_array))
            m2 = next_gen_fitness.index(min(next_gen_fitness))

            next_gen.append(pop[m1])
            next_gen.append(next_pop[m2])

            pop.remove(pop[m1])
            next_pop.remove(next_pop[m2])
            fitness_array.remove(min(fitness_array))
            next_gen_fitness.remove(min(next_gen_fitness))

        while len(next_gen) < pip_size:
            m1 = next_gen_fitness.index(min(next_gen_fitness))
            next_gen.append(next_pop[m1])
            next_pop.remove(next_pop[m1])

        gen_fitness = self.pop_fitness(next_gen, points)
        best_val_old_pop = min(gen_fitness)
        best_chromosome = next_gen[gen_fitness.index(min(gen_fitness))]

        return next_gen, best_val_old_pop, best_chromosome


# class Point():
#     def __init__(self, x, y):
#         self.x = float(x)
#         self.y = float(y)

algorithm = GeneticAlgorithm(
        pop_size=500,
        generations=500,
        p_crossover=0.05,
        p_mutation=0.05,
        dependency_factor=0.5
    )

def run_test_case(test_case):
        population = algorithm.create_population(test_case.degree)
        min_val, min_chromosome = math.inf, []

        for y in trange(algorithm.generations, ascii=True, desc='Test %s' % test_case.idx):
            (population, value, chromosome) = algorithm.genetic_algorithm(
                population, test_case.points, y)

            if value < min_val:
                min_val, min_chromosome = value, chromosome
                # y_calculated = ff.calculate_y(min_chromosome, test_case.points)
                # test_case.graph.update_pred(x_axis, y_calculated)

        # test_case.graph.save('graphs/case %s.png' % test_case.idx)
        return min_chromosome, min_val

def main():
    test_cases = []
    with open('input_examples.txt', 'r') as f:
        num_tests = int(f.readline())
        for i in range(num_tests):
            (n, d) = f.readline().split()
            n, degree = int(n), int(d)

            test_case = TestCase(n, degree, i, algorithm)
            for _ in range(n):
                (x, y) = f.readline().split()
                test_case.points.append((float(x), float(y)))
            test_cases.append(test_case)

    # outfile2 = open('output2.txt', 'w')

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(run_test_case)(test_case)
                                         for test_case in tqdm(test_cases, desc='Total', ascii=True))

    # outfile = open('output.txt', 'w')
    # for i, test in enumerate(test_cases):
    #     min_chromosome, min_val = test(algorithm)

    #     outfile.write('Case: %d \n' % (i + 1))
    #     for f in min_chromosome:
    #         outfile.write(str(f) + ' ')
    #     outfile.write(" value: %f \n" % min_val)
    #     print(" value: %f" % min_val)

    with open('output.txt', 'w') as f:
        for i, result in enumerate(results):
            min_chromosome, min_val = result
            
            graph = Graph()
            x_axis = [i[0] for i in test_case.points]
            y_axis = [i[1] for i in test_case.points]
            graph.update_org(x_axis, y_axis)
            y_calculated = ff.calculate_y(min_chromosome, test_case.points)
            graph.update_pred(x_axis, y_calculated)
            graph.save('graphs/case %s.png' % i)

            f.write('Case: %s\n' % (i + 1))
            for c in min_chromosome:
                f.write('%s ' % c)
            f.write(' value: %s\n' % min_val)


if __name__ == '__main__':
    main()
