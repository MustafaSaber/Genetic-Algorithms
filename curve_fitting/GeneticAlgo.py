from Object import Object
import random
import bisect
import FitnessFunction as ff
import matplotlib.pyplot as plt
from tqdm import trange
from joblib import Parallel, delayed
import multiprocessing

pop_size = 400
# Maximum number of generations
max_generations = 1000

# Probability of crossover between [ 0.4 , 0.7 ]
p_crossover = 0.65

# Probability of mutation between [ 0.001 , 0.1 ]
p_mutation = 1/pop_size

dependency_factor = 0.5


def create_population(degree):
    return [[random.uniform(-10, 10) for _ in range(degree+1)] for _ in range(pop_size)]


def mutate(chromosome, generation_number):
    for i in range(0, len(chromosome)):
        r1, r2, r3 = random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)
        val = -10 if r1 <= 0.5 else 10
        power = (1 - generation_number / max_generations) ** dependency_factor
        delta = val * (1 - r2 ** power)
        if r3 <= p_mutation:
            chromosome[i] = chromosome[i] - delta if r1 <= 0.5 else chromosome[i] + delta


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
        (Offspring1, Offspring2) = cross_over(pop[idx1], pop[idx2])
        mutate(Offspring1, generation_number)
        mutate(Offspring2, generation_number)
        next_pop.append(Offspring1)
        next_pop.append(Offspring2)

    next_gen_fitness = pop_fitness(next_pop, points)
    pip_size = len(pop)
    # Take best of previous and best of new.
    next_gen = []
    for i in range(len(pop)//2):
        m1, m2 = fitness_array.index(min(fitness_array)), next_gen_fitness.index(min(next_gen_fitness))
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


def main():
    # n = []
    # d = []
    # x = []
    # y = []
    # # Read all input from file
    # with open('input.txt', 'r') as infile:
    #     t = int(infile.readline())
    #     for i in range(t):
    #         (n1, n2) = infile.readline().split()
    #         n.append(int(n1))
    #         d.append(int(n2))
    #         x.append([])
    #         y.append([])
    #         for j in range(n[i]):
    #             a, b = infile.readline().split()
    #             x[i].append(a)
    #             y[i].append(b)
    #
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


    plt.ion()
    fig = plt.figure()
    plt.axis([-1, 50, -1, 50])
    ax = fig.add_subplot(111)
    pred_line, = ax.plot([], [], linestyle='-.')
    org_line, = ax.plot([], [], '-b')
    

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
        org_line.set_xdata(x_axis)
        org_line.set_ydata(y_axis)
        
        population = create_population(degree)
        min_val, min_chromosome, count = 20000000000000000, [], 0

        for y in range(max_generations):
            (population, value, chromosome) = genetic_algorithm(population, points, y)
            count += 1
            if value < min_val:
                count = 0
                min_val, min_chromosome = value, chromosome
            if count == 250:
                break
            y_calculated = ff.calculate_y(min_chromosome, points)
            
            pred_line.set_xdata(x_axis)
            pred_line.set_ydata(y_calculated)
            fig.canvas.draw()
            # print(" value: %d" % max_val)
        outfile.write('Case: %d \n' % (i+1))

        for f in min_chromosome:
            outfile.write(str(f) + ' ')
        outfile.write(" value: %f \n" % min_val)
        print(" value: %f" % min_val)


        # plt.plot(x_axis, y_axis)
        # outfile2.write('case: %d \n' % (i+1))
        # y_calculated = ff.calculate_y(min_chromosome, points)
        # for z in range(len(points)):
        #    outfile2.write('%f %f\n' % (points[z].x , y_calculated[z]))

        # plt.plot(x_axis, y_calculated, linestyle='-.')
        # plt.show()

    outfile.close()
    infile.close()


if __name__ == '__main__':
    main()