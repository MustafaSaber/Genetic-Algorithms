# The fitness function is approxmi the only thing that change in the genetic algorithm
# that's why I put it in another file


def fitness(chromosome, all_objects, max_weight):
    total_value, total_weight, index = 0, 0, 0

    for i in chromosome:
        if i == 1:
            total_value += all_objects[index].value
            total_weight += all_objects[index].weight
        index += 1
    if total_weight > max_weight:
        return 0
    else:
        return total_value
