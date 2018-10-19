from Object import Object
import random
import bisect
import Knapsack.FitnessFunction as ff
import matplotlib.pyplot as plt
from tqdm import trange
import matplotlib.pyplot as plt
# The size of solutions, everyone of it will be a solution wih the size of all objects.
# I have 5 Items
# a sample from POP will be 0 1 0 1 0

# It says that we took the second and fourth item.4 1
POP_Size = 300
# Maximum number of generations
MAX_GENERATIONS = 275

all_objects = []

max_weight = 0

# Probability of crossover between [ 0.4 , 0.7 ]
P_crossover = 0.4

P_mutation = 0.1

def create_population(all_objects):
    return [[random.randint(0, 1) for i in range(0, len(all_objects))] for n in range(0, POP_Size)]


def mutate(solution):
    for i in range(0, len(solution)):
        r = random.uniform(0, 1)
        if r <= P_mutation:
            solution[i] = solution[i] ^ 1


# The probability of crossover will be before we call the function
def cross_over(Parent1, Parent2 , AllObjects):
    r1 = random.randint(1, len(AllObjects)-1)
    r2 = random.uniform(0, 1)
    parent1, parent2 = list(Parent1), list(Parent2)
    if r2 <= P_crossover:
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

def pop_fitness(Pop , AllObjects , MAX_Weight):
    return [ff.fitness(i, AllObjects, MAX_Weight) for i in Pop]

def genentic_Algorithm(pop , AllObjects , MAX_Weight):

    fitness_array = pop_fitness(pop , AllObjects , MAX_Weight)
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
        (Offspring1, Offspring2) = cross_over(pop[idx1], pop[idx2] , AllObjects)
        mutate(Offspring1)
        # mutate(Offspring2)
        next_pop.append(Offspring1)
        # next_pop.append(Offspring2)


    next_gen_fitness = pop_fitness(next_pop, AllObjects, MAX_Weight)
    pip_size = len(pop)
    #Take best of previous and best of new.
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

    gen_fitness = pop_fitness(next_gen , AllObjects , MAX_Weight)
    gen_fitness.sort(reverse=True)
    best_val_old_pop = gen_fitness[0]

    return next_gen, best_val_old_pop

def main():
    outfile = open("output.txt", 'w')
    infile = open('input.txt', 'r')
    T = int(infile.readline())
    cases = 1
    for i in trange(T, desc='Total'):
        N = int(infile.readline())
        MAX_Weight = int(infile.readline())
        all_objects =[]
        for item in range(N):
            (v, w) = infile.readline().split()
            all_objects.append(Object(int(w), int(v)))
        population = create_population(all_objects)

        counter = [i for i in range(1, MAX_GENERATIONS + 1)]

        values = []
        Value , maxv = 0 , 0
        for y in trange(MAX_GENERATIONS, desc='Test %s' % i):
            (population, Value) = genentic_Algorithm(population, all_objects, MAX_Weight)
            values.append(Value)
            if Value > maxv:
                maxv = Value
        # plt.plot(counter, values)
        # plt.xlabel('generation number')
        # plt.ylabel('fitness of generation')
        # plt.show()
        outfile.write('Case: %d , value: %d\n' % (cases, maxv))
        cases += 1

    outfile.close()
    infile.close()


if __name__ == '__main__':
    main()
