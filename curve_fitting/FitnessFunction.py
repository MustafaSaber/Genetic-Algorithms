# The fitness function is approxmi the only thing that change in the genetic algorithm
# that's why I put it in another file


def calculate_y(chromosome, points):
    """ Chromosome list has the current coefficient values
     the length of the list is it's degree"""
    y_calculated = []
    for point in points:
        y_temp = 0
        for j in range(len(chromosome)):
            y_temp += chromosome[j] * (point[0] ** j)
        y_calculated.append(y_temp)
    return y_calculated


def fitness(chromosome, points):
    """ Mean square error """
    y_calculated, total_sum = calculate_y(chromosome, points), 0
    for i in range(len(points)):
        total_sum += ((points[i][1] - y_calculated[i]) ** 2)
    return (1 / len(points)) * total_sum

