# The fitness function is approxmi the only thing that change in the genetic algorithm
# that's why I put it in another file


def calculate_y(chromosome, points):
    """ chromosome list has the current coefficient values
     the length of the list is it's degree"""
    y_calculated = []
    for i in range(len(points)):
        y_temp = chromosome[0]
        for j in range(1, len(chromosome)):
            y_temp += chromosome[j] * (points[i].x ** j)
        y_calculated.append(y_temp)
    return y_calculated


def fitness(chromosome, points):
    y_calculated, total_sum = calculate_y(chromosome, points), 0
    for i in range(len(points)):
        total_sum += ((points[i].y - y_calculated[i]) ** 2)
    return (1/len(points)) * total_sum

