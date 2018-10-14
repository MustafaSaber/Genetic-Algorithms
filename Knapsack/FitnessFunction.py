# The fitness function is approxmi the only thing that change in the genetic algorithm
# that's why I put it in another file
def fitness(chromosome , AllObjects , MAX_Weight):
    total_value, total_weight, index = 0, 0, 0

    for i in chromosome:
        if i == 1:
            total_value += AllObjects[index].value
            total_weight += AllObjects[index].weight
        index += 1
    if total_weight > MAX_Weight:
        return 0
    else:
        return total_value