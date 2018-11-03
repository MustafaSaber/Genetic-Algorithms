from Object import Object
import random
import bisect
import FitnessFunction as ff
import matplotlib.pyplot as plt
from tqdm import trange
from joblib import Parallel, delayed
import multiprocessing
from graph import Graph
import math

pop_size = 20
# Maximum number of generations
max_generations = 1000

# Probability of crossover between [ 0.4 , 0.7 ]
p_crossover = 0.5

# Probability of mutation between [ 0.001 , 0.1 ]
p_mutation = 0.1

dependency_factor = 0.5


def create_population(degree):
    return [[random.uniform(-10, 10) for i in range(degree + 1)] for _ in range(pop_size)]


def mutate(chromosome, generation_number):
    power = (1 - generation_number/max_generations) ** dependency_factor
    for i in range(len(chromosome)):
        r3 = random.uniform(0, 1)
        if r3 <= p_mutation:
            r1, r2 = random.uniform(0, 1), random.uniform(0, 1)
            change = math.exp(-i)
            # val = (chromosome[i] + change) if r1 <= 0.5 else (chromosome[i] - change)
            delta = change * (1 - r2 ** power)
            if chromosome[i] + delta > 10:
                chromosome[i] -= delta
            elif chromosome[i] - delta < -10:
                chromosome[i] += delta
            else:
                chromosome[i] = chromosome[i] - delta if r1 <= 0.5 else chromosome[i] + delta


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
    fitness_array_cumlative = cumulative_sum(fitness_array)
    summation = fitness_array_cumlative[len(fitness_array_cumlative) - 1]

    # Proof that the value converges
    # print("%f && %f " % (min(fitness_array), max(fitness_array)))

    next_pop = []
    for _ in range(pop_size):
        r1, r2 = random.uniform(0, summation), random.uniform(0, summation)
        idx1 = bisect.bisect_left(fitness_array_cumlative, r1)
        idx2 = bisect.bisect_left(fitness_array_cumlative, r2)

        Offspring1, Offspring2 = cross_over(pop[idx1], pop[idx2])

        mutate(Offspring1, generation_number)
        mutate(Offspring2, generation_number)
        next_pop.append(Offspring1)
        next_pop.append(Offspring2)

    # Merge the new and old populations and take the best of both
    pop = pop + next_pop
    pop.sort(key=lambda x: ff.fitness(x, points), reverse=True)
    next_gen = pop[:pop_size]

    best_chromosome = next_gen[0]
    best_val_old_pop = ff.fitness(best_chromosome, points)

    return next_gen, best_val_old_pop, best_chromosome


def main():
    # num_cores = multiprocessing.cpu_count()
    # results = Parallel(n_jobs=num_cores)(delayed(run_testcase)(n[i], d[i], x[i], y[i])
    #                      for i in trange(t, desc='Total', ascii=True))
    #
    # with open("output.txt", 'w') as outfile:
    #     for case, val, chromosome in enumerate(results):
    #         outfile.write('Case: %d \n' % (i+1))
    #         for f in chromosome:
    #             outfile.write(str(f) + ' ')
    #         outfile.write(" value: %d \n" % val)


    infile = open('input_examples.txt', 'r')
    outfile = open('output.txt', 'w')
    # outfile2 = open('output2.txt', 'w')

    graph = Graph()

    test_cases = int(infile.readline())
    for i in range(test_cases):
        (n, d) = infile.readline().split()
        n_points, degree = int(n), int(d)

        points = []
        for j in range(n_points):
            (x, y) = infile.readline().split()
            points.append(Object(float(x), float(y)))

        x_axis = [i.x for i in points]
        y_axis = [i.y for i in points]
        graph.update_org(x_axis, y_axis)

        population = create_population(degree)
        max_val, max_chromosome, count = 0.0, [], 0

        for y in range(max_generations):
            (population, value, chromosome) = genetic_algorithm(population, points, y)
            count += 1
            if value > max_val:
                count = 0
                max_val, max_chromosome = value, chromosome
                y_calculated = ff.calculate_y(max_chromosome, points)
                graph.update_pred(x_axis, y_calculated)
                # print(max_chromosome)
            print('%s / %s: %s' % (y, max_generations, max_chromosome), end='\r')

            # if count == 250:
            #     break
            # print(" value: %d" % max_val)
        print('%s / %s: %s' % (y, max_generations, max_chromosome))
        
        outfile.write('Case: %d \n' % (i+1))
        for f in max_chromosome:
            outfile.write(str(f) + ' ')
        outfile.write(" value: %f \n" % max_val)
        print(" value: %f" % max_val)
        x_axis = [i.x for i in points]
        y_axis = [i.y for i in points]
        plt.plot(x_axis, y_axis, 'ro')
        # outfile2.write('case: %d \n' % (i+1))
        y_calculated = ff.calculate_y(max_chromosome, points)
        # for z in range(len(points)):
        #    outfile2.write('%f %f\n' % (points[z].x , y_calculated[z]))
        plt.plot(x_axis, y_calculated, linestyle='-.')
        plt.show()
    
    outfile.close()


if __name__ == '__main__':
    main()
