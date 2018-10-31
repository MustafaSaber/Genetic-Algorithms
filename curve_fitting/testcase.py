from graph import Graph
import math
from tqdm import trange
import FitnessFunction as ff

class TestCase():
    def __init__(self, n, degree, idx, algorithm):
        self.n = n
        self.degree = degree
        self.idx = idx
        self.points = []
        # self.graph = Graph()