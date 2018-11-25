class Set:
    def __init__(self, name, points=[]):
        self.name = name
        self.points = points

    def __str__(self):
        return '(' + str(self.name) + ', ' + str(self.points) + ')'
