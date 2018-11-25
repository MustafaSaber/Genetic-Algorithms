class Variable:
    def __init__(self, name, value, sets=[]):
        self.name = name
        self.value = value
        self.sets = sets

    def __str__(self):
        return '(' + str(self.name) + ', ' + str(self.value) + ', ' + str(self.sets) + ')'
