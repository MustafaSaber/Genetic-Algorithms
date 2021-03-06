from Knapsack.Object import Object
import random
import bisect
import Knapsack.FitnessFunction as ff
import matplotlib.pyplot as plt
from tqdm import trange
from joblib import Parallel, delayed
import multiprocessing

# The size of solutions, everyone of it will be a solution wih the size of all objects.
# I have 5 Items
# a sample from POP will be 0 1 0 1 0

# It says that we took the second and fourth item.4 1
pop_size = 300
# Maximum number of generations
max_generations = 400

# Probability of crossover between [ 0.4 , 0.7 ]
p_crossover = 0.5

# Probability of mutation between [ 0.001 , 0.1 ]
p_mutation = 0.05


def create_population(all_objects):
    return [[random.randint(0, 1) for _ in range(0, len(all_objects))] for _ in range(0, pop_size)]


def mutate(solution):
    for i in range(0, len(solution)):
        r = random.uniform(0, 1)
        if r <= p_mutation:
            solution[i] = solution[i] ^ 1


# The probability of crossover will be before we call the function
def cross_over(parent1, parent2, all_objects):
    r1 = random.randint(1, len(all_objects)-1)
    r2 = random.uniform(0, 1)
    parent1, parent2 = list(parent1), list(parent2)
    if r2 <= p_crossover:
        child1 = parent1[:r1] + parent2[r1:]
        child2 = parent2[:r1] + parent1[r1:]
        return child1, child2
    else:
        return parent1, parent2


# Will be used in selection
def cumulative_sum(fitness_list):
    new = []
    total_sum = 0
    for i in fitness_list:
        total_sum += i
        new.append(total_sum)
    return new


def pop_fitness(pop, all_objects, max_weight):
    return [ff.fitness(i, all_objects, max_weight) for i in pop]


def genetic_algorithm(pop, all_objects, max_weight):

    fitness_array = pop_fitness(pop, all_objects, max_weight)
    fitness_array_cumlative = cumulative_sum(fitness_array)
    summation = fitness_array_cumlative[len(fitness_array_cumlative) - 1]

    # counter = [i for i in range(1, len(Pop)+1)]
    # plt.plot(counter, fitness_array)
    # plt.xlabel('chromosome')
    # plt.ylabel('fitness of chromosome')
    # plt.show()

    next_pop = []
    while len(next_pop) < len(pop):
        r1, r2 = random.randint(0, summation), random.randint(0, summation)
        idx1 = bisect.bisect_left(fitness_array_cumlative, r1)
        idx2 = bisect.bisect_left(fitness_array_cumlative, r2)
        (Offspring1, Offspring2) = cross_over(pop[idx1], pop[idx2], all_objects)
        mutate(Offspring1)
        mutate(Offspring2)
        next_pop.append(Offspring1)
        next_pop.append(Offspring2)

    next_gen_fitness = pop_fitness(next_pop, all_objects, max_weight)
    pip_size = len(pop)
    # Take best of previous and best of new.
    next_gen = []
    for i in range(len(pop)//2):
        m1, m2 = fitness_array.index(max(fitness_array)), next_gen_fitness.index(max(next_gen_fitness))
        next_gen.append(pop[m1])
        next_gen.append(next_pop[m2])
        pop.remove(pop[m1])
        next_pop.remove(next_pop[m2])
        fitness_array.remove(max(fitness_array))
        next_gen_fitness.remove(max(next_gen_fitness))

    while len(next_gen) < pip_size:
        m1 = next_gen_fitness.index(max(next_gen_fitness))
        next_gen.append(next_pop[m1])
        next_pop.remove(next_pop[m1])

    gen_fitness = pop_fitness(next_gen, all_objects, max_weight)
    best_val_old_pop = max(gen_fitness)

    return next_gen, best_val_old_pop


def run_testcase(n, max_weight, v, w, test_num):
    all_objects = []
    for j in range(n):
        all_objects.append(Object(int(w[j]), int(v[j])))
    population = create_population(all_objects)

    # counter = [i for i in range(1, MAX_GENERATIONS + 1)]

    values = []
    value, max_value = 0, 0
    for y in range(max_generations):
        (population, value) = genetic_algorithm(population, all_objects, max_weight)
        values.append(value)
        if value > max_value:
            max_value = value
        # plt.plot(counter, values)
        # plt.xlabel('generation number')
        # plt.ylabel('fitness of generation')
        # plt.show()
        # outfile.write('Case: %d , value: %d\n' % (cases, maxv))
    return max_value


def main():
    # outfile = open("output.txt", 'w')

    n = []
    max_weight = []
    v = []
    w = []

    # Read all input from file
    with open('input.txt', 'r') as infile:
        t = int(infile.readline())
        for i in range(t):
            n.append(int(infile.readline()))
            max_weight.append(int(infile.readline()))
            v.append([])
            w.append([])
            for j in range(n[i]):
                a, b = infile.readline().split()
                v[i].append(a)
                w[i].append(b)

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(run_testcase)(n[i], max_weight[i], v[i], w[i], i)
                         for i in trange(t, desc='Total', ascii=True))
    
    with open("output.txt", 'w') as outfile:
        for case, val in enumerate(results):
            outfile.write('Case: %d , value: %d\n' % (case + 1, val))


if __name__ == '__main__':
    main()
