import ga
import random
import visualization
import multiprocessing
import matplotlib.pyplot as plt

plt.ion()
POP_SIZE = 100
_COUNT = 100
SIZE = 100
city_coords = [(random.randint(0, SIZE), random.randint(0, SIZE)) for _ in range(_COUNT)]
gen_fitness_mmm = {}


def dist(i: int, j: int):
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
        j = random.randint(0, len(self.__values) - 1)
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
        length = random.randint(0, _COUNT - 1 - len(a) // 7)
        x = random.randint(0, _COUNT - length - 1)
        x_new = random.randint(0, _COUNT - 1)

        result = []
        for bi in range(x_new):
            if b[bi] not in a[x:(x + length)]:
                result.append(b[bi])
        result.extend(a[x:(x + length)])
        for bi in range(x_new, _COUNT):
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


def add_data(_generation, fitnesses, _best_member, hp, _f, *_):
    if hp not in gen_fitness_mmm:
        gen_fitness_mmm[hp] = [[], [], []]
    gen_fitness_mmm[hp][0].append(fitnesses[0])
    gen_fitness_mmm[hp][1].append(fitnesses[1])
    gen_fitness_mmm[hp][2].append(fitnesses[2])

    for hp, value in gen_fitness_mmm.items():
        min_values = value[0]
        if len(min_values) > 3 and min_values[-1] != min_values[-2]:
            print(min_values[-1])
        break




def plot_quartiles(subplot, _generation, _fitnesses, _best_member, _hp, *_):
    subplot.set_yscale('log')
    subset_size = 300
    for hp_, gen_fitness_mmm_ in gen_fitness_mmm.items():
        for data in gen_fitness_mmm_:
            y = list(data[(max(0, len(data) - subset_size )):])
            x = list(i + max(0, len(data) - subset_size) for i in range(len(y)))
            subplot.scatter(x, y, label=hp_)
    subplot.legend(loc=(0.90, 0.9))


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def plot(subplot, _generation, _fitnesses, best_member, *_):
    # print('plotting gen', _generation)
    x = [city_coords[i][0] for i in best_member]
    y = [city_coords[i][1] for i in best_member]
    subplot.plot(x, y, '-o', marker=" ")

def plot_median(subplot, _generation, _fitnesses, best_member, _hp, _f, median_member):
    # print('plotting gen', _generation)
    x = [city_coords[i][0] for i in median_member]
    y = [city_coords[i][1] for i in median_member]
    subplot.plot(x, y, '-o', marker=" ")


f  = [[] for i in range(POP_SIZE)]
def plot_members(subplot, _generation, _fitnesses, best_member, _hp, fitnesses, *_):
    for i in range(len(fitnesses)):
        val = fitnesses[i]
        f[i].append(val)
        subplot.plot(f[i])


q = multiprocessing.Queue()


def _worker(q_, city_coords_from_other_thread):
    global city_coords
    city_coords = city_coords_from_other_thread
    for crossover_fraction in {0.8}:
        for mutation_fraction in {0.1}:
            ga.ga(POP_SIZE,
                  Member.fitness,
                  Member.create_random,
                  Member.mutate,
                  Member.crossover,
                  Member.clone,
                  mutation_fraction=mutation_fraction,
                  crossover_fraction=crossover_fraction,
                  callbacks=[_data_pruner(q_, (mutation_fraction, crossover_fraction))],
                  max_generation=-1
                  )


if __name__ == '__main__':
    process = multiprocessing.Process(target=_worker, args=(q, city_coords))
    process.start()

    visualization.loop(q, {(1, 1): plot,
                           (1, 2): plot_median,
                           (1, 3): plot_quartiles}, add_data)

# I've gone through all these hoops because of two things:
# 1. It's impossible to share large amounts (i.e. MBs already) between processes and threads in python using the Queue
# The Queue._buffer just accumulates and only one is withdrawn every time a get is called
# Because the _buffer is on the putting side of the pipe, I conclude that either I'm doing something wrong regarding GIL
# or the pipe is excruciatingly slow. I think it's the latter actually, weirdly enough:
# Putting the fitness summary over the wire rather than the fitness list itself solved the problem. Big red flag.
# 2. It's impossible to make the UI perfectly responsive on a secondary process. Sometimes it feels okay, but every once
# in a while it is just sluggish or it hangs. plt is supposed to be run on a single thread, so I should stick to main


class _data_pruner:
    def __init__(self, q_, hp):
        self.__q = q_
        self.__hp = hp

    def __call__(self, generation, population, fitnesses):
        if generation % 100 == 0:
            fitness_summary = [fitnesses[0], mean(fitnesses), fitnesses[-1]]
            self.__q.put((generation,
                          fitness_summary,
                          population[0].values,
                          self.__hp,
                          [], #fitnesses
                          population[POP_SIZE // 2].values))



