from Knapsack.Object import Object
import random
import bisect
import Knapsack.FitnessFunction as ff
import matplotlib.pyplot as plt
# The size of solutions, everyone of it will be a solution wih the size of all objects.
# I have 5 Items
# a sample from POP will be 0 1 0 1 0

# It says that we took the second and fourth item.
POP_Size = 5
# Maximum number of generations
MAX_GENERATIONS = 10

# Probability of crossover between [ 0.4 , 0.7 ]
P_crossOver = 0.6

# Probability of mutation between [ 0.001 , 0.1 ]
P_Mutation = 0.1


def CreatePopulation(AllObjects):
    return [[random.randint(0, 1) for i in range(0, len(AllObjects))] for n in range(0, POP_Size)]


def mutate(solution):
    for i in range(0, len(solution)):
        r = random.uniform(0, 1)
        if r <= P_Mutation:
            solution[i] = solution[i]^1


# The probability of crossover will be before we call the function
def CrossOver(Parent1, Parent2 , AllObjects):
    r1 = random.randint(1, len(AllObjects)-1)
    r2 = random.uniform(0, 1)
    Parent1, Parent2 = list(Parent1), list(Parent2)
    if r2 <= P_crossOver:
        child1 = Parent1[:r1] + Parent2[r1:]
        child2 = Parent2[:r1] + Parent1[r1:]
        return child1, child2
    else:
        return Parent1, Parent2


# Will be used in selection
def CumulativeSum(list):
    New = []
    total_sum = 0
    for i in list:
        total_sum += i
        New.append(total_sum)
    return New


def Population_fitness(Pop , AllObjects , MAX_Weight):
    return [ff.fitness(i, AllObjects, MAX_Weight) for i in Pop]

def genentic_Algorithm(Pop , AllObjects , MAX_Weight):

    fitness_array = Population_fitness(Pop , AllObjects , MAX_Weight)
    fitness_array_cumlative = CumulativeSum(fitness_array)
    summation = fitness_array_cumlative[len(fitness_array_cumlative) - 1]

    counter = [i for i in range(1, len(Pop)+1)]
    plt.plot(counter, fitness_array)
    plt.xlabel('chromosome')
    plt.ylabel('fitness of chromosome')
    plt.show()

    NextPop = []
    while len(NextPop) < len(Pop):
        r1, r2 = random.randint(0, summation), random.randint(0, summation)
        idx1 = bisect.bisect_left(fitness_array_cumlative, r1)
        idx2 = bisect.bisect_left(fitness_array_cumlative, r2)
        (Offspring1, Offspring2) = CrossOver(Pop[idx1], Pop[idx2] , AllObjects)
        mutate(Offspring1)
        mutate(Offspring2)
        NextPop.append(Offspring1)
        NextPop.append(Offspring2)

    fitness_array.sort(reverse=True)
    BestValueOldPop = fitness_array[0]

    return NextPop, BestValueOldPop

def main():
    file = open("output.txt", 'w')
    T = int(input("Enter number of Test cases: "))
    cases = 1
    for i in range(T):
        N = int(input("Enter number of knapsack items: "))
        MAX_Weight = int(input("Enter max value of knapsack: "))
        AllObjects =[]
        for item in range(N):
            (v, w) = input().split()
            AllObjects.append(Object(int(w), int(v)))
        Population = CreatePopulation(AllObjects)

        counter = [i for i in range(1, MAX_GENERATIONS + 1)]

        values = []
        Value , maxv = 0 , 0
        for y in range(0, MAX_GENERATIONS):
            (Population, Value) = genentic_Algorithm(Population, AllObjects, MAX_Weight)
            values.append(Value)
            if Value > maxv:
                maxv = Value
        plt.plot(counter, values)
        plt.xlabel('generation number')
        plt.ylabel('fitness of generation')
        plt.show()
        file.write('Case: %d , value: %d\n' % (cases, maxv))
        cases+=1

    file.close()


if __name__ == '__main__':
    main()