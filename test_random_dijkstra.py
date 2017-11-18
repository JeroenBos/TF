import unittest
from random_dijkstra import *


class dijkstra(unittest.TestCase):
    def test_simple(self):
        c = Dijkstra([0], lambda current: [current + 1], lambda current: current == 5, lambda x: x)

        result = list(next(iter(c.find_route())))

        self.assertSequenceEqual(result, [5, 4, 3, 2, 1, 0])

    def test_dual(self):
        c = Dijkstra([(0, 0)],
                     lambda current: [(current[0] + 1, random.randint(0, 10)), (current[0] + 1, random.randint(0, 10))],
                     lambda current: current[0] == 5,
                     lambda x: x[0])

        result = list(c[0] for c in next(iter(c.find_route())))

        self.assertSequenceEqual(result, [5, 4, 3, 2, 1, 0])

