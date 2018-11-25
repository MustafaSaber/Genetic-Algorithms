from numpy.distutils.fcompiler import none
import copy
from fuzzy_logic.variable import Variable
from fuzzy_logic.set import Set
from fuzzy_logic.point import Point


def intersect(val, p1, p2):
    def slope(p_1, p_2):
        return 0.0 if (p_1.x - p_2.x == 0) else ((p_2.y - p_1.y) / (p_2.x - p_1.x))

    return (slope(p1, p2) * (val - p1.x)) + p1.y


def fuzzification(all_variables):
    d = {}
    for i in range(len(all_variables)):
        for j in range(len(all_variables[i].sets)):
            d[all_variables[i].name + all_variables[i].sets[j].name] = 0
            for z in range(0, len(all_variables[i].sets[j].points) - 1):
                if all_variables[i].sets[j].points[z].x <= all_variables[i].value <= \
                        all_variables[i].sets[j].points[z + 1].x:
                    
                    y = intersect(all_variables[i].value,
                                  all_variables[i].sets[j].points[z], all_variables[i].sets[j].points[z + 1])
                    d[all_variables[i].name + all_variables[i].sets[j].name] = y
                    break
    return d


def inference(r, fuzzification_dict):
    for i in range(len(r)):
        new_list = copy.deepcopy(r[i])
        for j in range(len(r[i])):
            if r[i][j] == 'and':
                if r[i][j - 1] == 'not':
                    new_list[j] = 1 - min(fuzzification_dict[r[i][j - 3] + r[i][j - 2]],
                                          fuzzification_dict[r[i][j + 1] + r[i][j + 2]])
                    new_list.remove(r[i][j - 3]), new_list.remove(r[i][j - 2]), new_list.remove(r[i][j + 1])
                    new_list.remove(r[i][j + 1]), new_list.remove(r[i][j - 1])
                else:
                    new_list[j] = min(fuzzification_dict[r[i][j - 2] + r[i][j - 1]],
                                      fuzzification_dict[r[i][j + 1] + r[i][j + 2]])
                    new_list.remove(r[i][j - 2]), new_list.remove(r[i][j - 1]), new_list.remove(r[i][j + 1])
                    new_list.remove(r[i][j + 2])
        r[i] = new_list

    for i in range(len(r)):
        new_list = copy.deepcopy(r[i])
        for j in range(len(r[i])):
            if r[i][j] == 'or':
                if r[i][j - 1] == 'not':
                    new_list[j] = 1 - max(fuzzification_dict[r[i][j - 3] + r[i][j - 2]],
                                          fuzzification_dict[r[i][j + 1] + r[i][j + 2]])
                    new_list.remove(r[i][j - 3]), new_list.remove(r[i][j - 2]), new_list.remove(r[i][j + 1])
                    new_list.remove(r[i][j + 2]), new_list.remove(r[i][j - 1])
                else:
                    new_list[j] = max(fuzzification_dict[r[i][j - 2] + r[i][j - 1]],
                                      fuzzification_dict[r[i][j + 1] + r[i][j + 2]])
                    new_list.remove(r[i][j - 2]), new_list.remove(r[i][j - 1]), new_list.remove(r[i][j + 1])
                    new_list.remove(r[i][j + 2])
        r[i] = new_list

    return r


def centroid(output_var):
    d = {}
    for current_set in output_var.sets:
        a = 0
        for i in range(len(current_set.points)-1):
            a += ((current_set.points[i].x * current_set.points[i + 1].y) -
                  (current_set.points[i + 1].x * current_set.points[i].y))
        a /= 2

        cx = 0
        for i in range(len(current_set.points)-1):
            cx += (current_set.points[i].x + current_set.points[i + 1].x) *\
                  ((current_set.points[i].x * current_set.points[i + 1].y) -
                   (current_set.points[i + 1].x * current_set.points[i].y))
        cx /= (6 * a)

        d[current_set.name] = cx

    return d


def defuzzification(rules_modifies, centroid_values):
    numerator = dominator = 0

    for l in rules_modifies:
        numerator += l[0] * centroid_values[l[len(l) - 1]]
        dominator += l[0]

    return numerator/dominator


def get_input():
    all_variables = []
    with open("input.txt", 'r') as infile:
        var_num = int(infile.readline())
        # input variables (names, values) and sets
        for _ in range(var_num):
            name, value = infile.readline().split()
            all_variables.append(Variable(str(name).lower(), float(value), []))
            num_of_sets = int(infile.readline())
            for _ in range(num_of_sets):
                name, poly = infile.readline().split()
                points = infile.readline().split()
                points = list(map(float, points))
                points = [Point(points[i], 0) if (i == 0 or i == len(points) - 1) else Point(points[i], 1)
                          for i in range(len(points))]
                current_set = Set(name.lower(), points)
                all_variables[len(all_variables) - 1].sets.append(current_set)
                del current_set

        # output variable name and sets
        name = infile.readline()
        output_variable = Variable(str(name).lower(), 0)
        num_of_sets = int(infile.readline())
        for _ in range(num_of_sets):
            name, poly = infile.readline().split()
            points = infile.readline().split()
            points = list(map(float, points))
            points = [Point(points[i], 0) if (i == 0 or i == len(points) - 1) else Point(points[i], 1)
                      for i in range(len(points))]
            temp_set = Set(name.lower(), points)
            output_variable.sets.append(temp_set)
            del temp_set

        rules_num = int(infile.readline())
        rules = []
        for _ in range(rules_num):
            line = infile.readline().split()
            # num_of_premises = int(line[0])
            d, coun, j = [], 1, 1
            for j in range(1, len(line)):
                if line[j] == 'then' or line[j] == '=':
                    continue
                d.append(line[j].lower())
            rules.append(d)

    return all_variables, output_variable, rules


def main():
    all_variables, output_variable, rules = get_input()
    f = fuzzification(all_variables)
    r = inference(rules, f)
    centroid_dict = centroid(output_variable)
    with open('output.txt', 'w') as outfile:
        outfile.write("Fuzzification: \n\n")
        for i in f:
            outfile.write(str(i) + ' ' + str(f[i]) + '\n')
        outfile.write('\n')
        outfile.write("Inference of rules: \n\n")
        for i in r:
            outfile.write(str(i) + '\n')
        outfile.write('\n')
        outfile.write("Defuzzification: ")
        outfile.write(str(defuzzification(r, centroid_dict)))
    print()


if __name__ == "__main__":
    main()
