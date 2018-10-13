class Object:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight
    def DisplayItem(self):
        print("Value: ", self.value, "\nWeight: ", self.weight)