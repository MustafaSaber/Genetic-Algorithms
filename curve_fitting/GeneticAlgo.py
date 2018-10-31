from Object import Object
import random
import bisect
import FitnessFunction as ff
import matplotlib.pyplot as plt
from tqdm import trange
from joblib import Parallel, delayed
import multiprocessing
import math
from graph import Graph
import time


pop_size = 100
# Maximum number of generations
max_generations = 100

# Probability of crossover between [ 0.4 , 0.7 ]
p_crossover = 0.7

# Probability of mutation between [ 0.001 , 0.1 ]
p_mutation = 0.1

dependency_factor = 0.5


def create_population(degree):
    return [[random.uniform(-10, 10) for _ in range(degree + 1)] for _ in range(pop_size)]


def mutate(chromosome, generation_number):
    for i in range(0, len(chromosome)):
        r1, r2, r3 = random.uniform(0, 1), random.uniform(
            0, 1), random.uniform(0, 1)
        val = -10 if r1 <= 0.5 else 10
        power = (1 - generation_number / max_generations) ** dependency_factor
        delta = val * (1 - r2 ** power)
        if r3 <= p_mutation * (1 - generation_number / max_generations):
            chromosome[i] = random.uniform(-10, 10)


# The probability of crossover will be before we call the function
def cross_over(parent1, parent2):
    r1 = random.randint(1, len(parent1) - 1)
    r2 = random.uniform(0, 1)
    parent1, parent2 = list(parent1), list(parent2)
    if r2 <= p_crossover:
        child1 = parent1[:r1] + parent2[r1:]
        child2 = parent2[:r1] + parent1[r1:]
        return child1, child2
    else:
        return parent1, parent2


def cross_over_non_uniform(parent1, parent2):
    child = [list(parent1), list(parent2)]
    for i in range(len(parent1)):
        r2 = random.uniform(0, 1)
        if r2 <= p_crossover:
            # Swap
            child[0][i], child[1][i] = child[1][i], child[0][i]
    return child[0], child[1]


# Will be used in selection
def cumulative_sum(fitness_list):
    new = []
    total_sum = 0
    for i in fitness_list:
        total_sum += i
        new.append(total_sum)
    return new


def pop_fitness(pop, points):
    return [ff.fitness(i, points) for i in pop]


def genetic_algorithm(pop, points, generation_number):

    fitness_array = pop_fitness(pop, points)
    # print(fitness_array)
    fitness_array_cumlative = cumulative_sum(fitness_array)
    summation = fitness_array_cumlative[len(fitness_array_cumlative) - 1]
    # Proof that the value converges
    # print("%f && %f " % (min(fitness_array), max(fitness_array)))
    next_pop = []
    while len(next_pop) < len(pop):
        r1, r2 = random.uniform(0, summation), random.uniform(0, summation)
        idx1 = bisect.bisect_left(fitness_array_cumlative, r1)
        idx2 = bisect.bisect_left(fitness_array_cumlative, r2)

        Offspring1, Offspring2 = cross_over_non_uniform(pop[idx1], pop[idx2])

        mutate(Offspring1, generation_number)
        mutate(Offspring2, generation_number)
        next_pop.append(Offspring1)
        next_pop.append(Offspring2)

    next_gen_fitness = pop_fitness(next_pop, points)
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

    gen_fitness = pop_fitness(next_gen, points)
    best_val_old_pop = min(gen_fitness)
    best_chromosome = next_gen[gen_fitness.index(min(gen_fitness))]

    # x_axis = [i.x for i in points]
    # for i, chromosome in enumerate(next_gen):
    #     if i == 10:
    #         break
    #     time.sleep(1)
    #     print(chromosome)
    #     y_calculated = ff.calculate_y(chromosome, points)
    #     graph.update_example(x_axis, y_calculated)

    return next_gen, best_val_old_pop, best_chromosome


def run_testcase(n, d, x, y):
    points = []
    for j in range(n):
        points.append(Object(float(x[j]), float(y[j])))
    population = create_population(d)
    max_chromosome, max_value = 0, 0
    for y in range(max_generations):
        (population, value, chromosome) = genetic_algorithm(population, points, y)
        if value > max_value:
            max_value, max_chromosome = value, chromosome
    return max_value, max_chromosome


class Point():
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)


class TestCase():
    def __init__(self, n, degree, idx):
        self.n = n
        self.degree = degree
        self.idx = idx
        self.points = []
        self.graph = Graph()

    def __call__(self):
        x_axis = [i.x for i in self.points]
        y_axis = [i.y for i in self.points]
        self.graph.update_org(x_axis, y_axis)
        population = create_population(self.degree)
        min_val, min_chromosome = math.inf, []

        for y in trange(max_generations, ascii=True, desc='Generation'):
            (population, value, chromosome) = genetic_algorithm(
                population, self.points, y)

            if value < min_val:
                min_val, min_chromosome = value, chromosome
                y_calculated = ff.calculate_y(min_chromosome, self.points)
                self.graph.update_pred(x_axis, y_calculated)

        self.graph.save('graphs/case %s.png' % self.idx)
        return min_chromosome, min_val


def main():
    # num_cores = multiprocessing.cpu_count()
    # results = Parallel(n_jobs=num_cores)(delayed(run_testcase)(n[i], d[i], x[i], y[i])
    #                      for i in trange(t, desc='Total', ascii=True))

    test_cases = []
    with open('input_examples.txt', 'r') as f:
        num_tests = int(f.readline())
        for i in range(num_tests):
            (n, d) = f.readline().split()
            n, degree = int(n), int(d)

            test_case = TestCase(n, degree, i)
            for _ in range(n):
                (x, y) = f.readline().split()
                test_case.points.append(Point(x, y))
            test_cases.append(test_case)

    outfile = open('output.txt', 'w')
    # outfile2 = open('output2.txt', 'w')

    for i, test in enumerate(test_cases):
        min_chromosome, min_val = test()

        outfile.write('Case: %d \n' % (i + 1))
        for f in min_chromosome:
            outfile.write(str(f) + ' ')
        outfile.write(" value: %f \n" % min_val)
        print(" value: %f" % min_val)

    outfile.close()


if __name__ == '__main__':
    main()
