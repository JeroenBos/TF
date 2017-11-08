class Member:
    def mutate(self):
        raise NotImplementedError('subclass must implement mutate')

    def clone(self):
        raise NotImplementedError('subclass must implement clone')

    def fitness(self):
        raise NotImplementedError('subclass must implement fitness')

    @staticmethod
    def crossover(a, b):
        raise NotImplementedError('subclass must implement crossover')

    @staticmethod
    def create_random():
        raise NotImplementedError('subclass must implement create_random')


# noinspection PyAbstractClass
class MemberWithCachedFitness(Member):
    def __init__(self):
        self.__cached_fitness = None

    def fitness(self):
        if self.__cached_fitness is None:
            self.__cached_fitness = self._fitness()

    def _fitness(self):
        raise NotImplementedError('subclass must implement _fitness')



