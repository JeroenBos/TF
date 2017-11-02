import ga
import random
import visualization
import itertools
import time
import multiprocessing
import queue
import matplotlib.pyplot as plt

plt.ion()
_COUNT = 20
SIZE = 100
city_coords = [(random.randint(0, SIZE), random.randint(0, SIZE)) for _ in range(_COUNT)]


def dist(i, j):
    c1 = city_coords[i % len(city_coords)]
    c2 = city_coords[j % len(city_coords)]
    d = (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2
    return d


class Member:
    def __init__(self, values):
        assert isinstance(values, list)
        assert len(values) == _COUNT
        self.__values = values

    def mutate(self):
        i = random.randint(0, len(self.__values) - 1)
        j = (i + 1) % _COUNT
        self.__values[i], self.__values[j] = self.__values[j], self.__values[i]

    def clone(self):
        return Member([x for x in self.__values])

    def fitness(self):
        return sum(dist(self.__values[i % _COUNT], self.__values[i + 1]) for i in range(_COUNT - 1))

    @staticmethod
    def crossover(a, b):
        a = a.__values
        b = b.__values

        # the order of a and b is random already, so we can just choose a wlog
        length = random.randint(0, _COUNT - 1)
        x = random.randint(0, _COUNT - length - 1)

        result = []
        for bi in range(x):
            if b[bi] not in a[x:(x + length)]:
                result.append(b[bi])
        result.extend(a[x:(x + length)])
        for bi in range(x, _COUNT):
            if b[bi] not in result:
                result.append(b[bi])

        assert len(set(result)) == _COUNT
        return Member(result)

    @staticmethod
    def create_random():
        result = list(range(_COUNT))
        random.shuffle(result)
        return Member(result)

    def __repr__(self):
        return 'member fitness = ' + str(self.fitness())

    @property
    def values(self):
        return self.__values


#gen_fitness_mmm = [[], [], []]
#prev = None
#prev_f = None

#def callback(_generation, population, fitnesses):
#    gen_fitness_mmm[0].append(fitnesses[0])
#    gen_fitness_mmm[1].append(mean(fitnesses))
#    gen_fitness_mmm[2].append(fitnesses[-1])
#    global prev, prev_f
#    if _generation % 500 == 0:
#        print('gen: ', _generation)
#        assert fitnesses[0] == min(fitnesses)
#        assert population[0].fitness() == fitnesses[0]
#        assert population[0].fitness() == gen_fitness_mmm[0][-1]
#
#    if prev:
#        if fitnesses[0] != prev_f:
#            assert population[0] != prev
#            assert population[0].values != prev.values
#        else:
#            assert population[0].fitness() == prev_f
#
#
#    prev = population[0]
#    prev_f = fitnesses[0]

gen_fitness_mmm = [[], [], []]
def plot_quartiles(subplot, _generation, fitnesses, _best_member):
    gen_fitness_mmm[0].append(fitnesses[0])
    subplot.set_yscale('log')
    L = min(100, len(gen_fitness_mmm[0]))
    subplot.scatter(list(range(L)), gen_fitness_mmm[0][-L:])


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def plot(subplot, _generation, _fitnesses, best_member):
    print('plotting gen', _generation)
    x = [city_coords[i][0] for i in best_member]
    y = [city_coords[i][1] for i in best_member]
    subplot.plot(x, y, '-o', marker=" ")


q = multiprocessing.Queue()


def _worker(q_):
    ga.ga(100,
          Member.fitness,
          Member.create_random,
          Member.mutate,
          Member.crossover,
          Member.clone,
          mutation_fraction=0,
          crossover_fraction=0.1,
          callbacks=[_data_pruner(q_)]
          )


if __name__ == '__main__':
    process = multiprocessing.Process(target=_worker, args=(q,))
    process.start()

    visualization.loop(q, {(1, 1): plot,
                           (1, 2): plot_quartiles})

# I've gone through all these hoops because of two things:
# 1. It's impossible to share large amounts (i.e. MBs already) between processes and threads in python using the Queue
# The Queue._buffer just accumulates and only one is withdrawn every time a get is called
# Because the _buffer is on the putting side of the pipe, I conclude that either I'm doing something wrong regarding GIL
# or the pipe is excruciatingly slow. I think it's the latter actually, weirdly enough:
# Putting the fitness summary over the wire rather than the fitness list itself solved the problem. Big red flag.
# 2. It's impossible to make the UI perfectly responsive on a secondary process. Sometimes it feels okay, but every once
# in a while it is just sluggish or it hangs. plt is supposed to be run on a single thread, so I should stick to main

class _data_pruner:
    def __init__(self, q_):
        self.__q = q_

    def __call__(self, generation, population, fitnesses):
        fitness_summary = [fitnesses[0], mean(fitnesses), fitnesses[-1]]
        self.__q.put((generation, fitness_summary, population[0].values))



