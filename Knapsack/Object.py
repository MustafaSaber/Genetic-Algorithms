class Object:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight

    def display_item(self):
        print("Value: ", self.value, "\nWeight: ", self.weight)
