import random
from itertools import count


def ga(fitness,
       generate,
       population_size,
       mutate,
       crossover,
       clone,
       mutation_fraction=0.05,
       crossover_fraction=1.0,
       kill_fraction=0.5,
       kill_std_dev=0.1,
       callbacks=None,
       max_generation=-1):
    """
    :param fitness: A function that takes a member and computes its fitness.
    :param generate: A function that takes nothing and generates a random member.
    :param population_size: The size of the population.
    :param mutate: A function that takes a member and returns it in a mutated form.
    :param crossover: A function that takes two members and returns a new combined member.
    :param clone: A function that deep clones a member.
    :param mutation_fraction: The fraction of the population that survives a generation that undergoes a mutation.
    :param crossover_fraction: The fraction of the killed population that is regenerated by crossover.
                               The remainder is regenerated by mutating clones of survivors.
    :param kill_fraction: The fraction of the population that is killed after each generation.
    :param kill_std_dev: The standard deviation causing some members above (below) the fill_fraction to survive (die).
    :param callbacks: A list of functions that each takes the generation index, the generation and their fitnesses.
    :param max_generation: The number of generations to simulate. -1 means indefinite.
    :return: The last generation.
    """

    assert max_generation >= 1 or max_generation == -1
    callbacks = callbacks or []
    if not mutate:
        mutation_fraction = 0
    if not crossover:
        crossover_fraction = 0

    population = [generate() for _ in range(population_size)]

    for generation_index in (range(max_generation) if max_generation != -1 else count(0, 1)):
        fitnesses = _compute_fitness(population, fitness)

        population, fitnesses = zip(*sorted(zip(population, fitnesses), key=lambda t: t[1]))

        survivors = _kill(population, kill_fraction, kill_std_dev)
        children = _crossover(population, survivors, crossover_fraction, crossover)
        clones = _regenerate(survivors, len(population) - len(children) - len(survivors), mutate, clone)
        _mutate(survivors, mutation_fraction, mutate)

        population = survivors + children + clones

        assert len(population) == len(fitnesses)
        for callback in callbacks:
            callback(generation_index, population, fitnesses)


def _compute_fitness(population, fitness):
    return [fitness(member) for member in population]  # TODO: parallelize


def _kill(population, kill_fraction, std_dev):
    kill_median = int(round(kill_fraction * len(population)))
    assert kill_median != len(population)
    scaled_std_dev = int(round(std_dev * len(population)))

    # get the indices of members that above (below) the threshold are (aren't) killed where they usually would (not) be
    exception_indices = normalintsample(0, len(population), len(population) - kill_median, scaled_std_dev, kill_median)

    survivors = [population[i] for i in set(range(0, kill_median)) ^ exception_indices]
    return survivors


def _mutate(population, mutation_fraction, mutate):
    if mutation_fraction == 0:
        return
    for member in population:
        if member is not None:
            if random.uniform(0, 1) < mutation_fraction:
                mutate(member)


def _crossover(population, survivors, fraction, crossover):
    if fraction == 0:
        return

    n = int(round(fraction * (len(population) - len(survivors))))
    return [crossover(*random.sample(survivors, 2)) for _ in range(n)]


def _regenerate(survivors, n, mutate, clone):
    result = [clone(random.sample(survivors, 1)[0]) for _ in range(n)]
    for member in result:
        mutate(member)
    return result


def _merge(gapped, fillings):
    """
    Fills the specified fillings at the positions in the gapped list where there is currently a None
    """
    i = 0
    for new in fillings:
        while gapped[i]:
            i += 1
            if len(gapped) == i:
                return
        gapped[i] = new
        i += 1
        if len(gapped) == i:
            return


def normalintsample(low, high, mean, std_dev, n):
    """
    Draws n unique number in [low, high) according to a normal distribution with parameters mean and std_dev
    """
    assert 0 <= n < high - low

    indices = set()
    while len(indices) != n:
        sample = int(round(random.gauss(mean, std_dev)))
        if low <= sample < high:
            indices.add(sample + low)
    return indices
