import ga
import random
from math import *
import visualization
import matplotlib.pyplot as plt

_X_SCALE = .2
_LENGTH = 100


class Member:
    def __init__(self, values):
        assert isinstance(values, list)
        self.__values = values

    def mutate(self):
        i = random.randint(0, len(self.__values) - 1)
        self.__values[i] = random.gauss(0, 1) * abs(self.__values[i])

    def clone(self):
        return Member([x for x in self.__values])

    def fitness(self):
        return sum((val - sin(i * _X_SCALE)) ** 2 for i, val in enumerate(self.__values))

    @property
    def values(self):
        return self.__values

    @staticmethod
    def crossover(a, b):
        return Member([[a, b][random.randint(0, 1)].__values[i] for i in range(_LENGTH)])

    @staticmethod
    def create_random():
        return Member([random.uniform(-1, 1) for _ in range(_LENGTH)])

    def __repr__(self):
        return 'member fitness = ' + str(self.fitness())


def callback(_generation, population, _fitnesses):
    visualization.plot(plot, pts=population[0].values)


def plot(pts):
    plt.scatter([range(_LENGTH)], pts)
    plt.scatter([range(_LENGTH)], [sin(i * _X_SCALE) for i in range(_LENGTH)])


if __name__ == '__main__':
    ga.ga(100,
          Member.fitness,
          Member.create_random,
          Member.mutate,
          Member.crossover,
          Member.clone,
          crossover_fraction=0.5,
          callbacks=[callback])
