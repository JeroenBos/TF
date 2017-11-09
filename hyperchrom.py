from typing import List, Tuple, Dict, Union
import keras
import itertools
import random
import collections


class Genome:
    def __init__(self, space, fitness, max_learnable_params=-1):
        assert isinstance(space, dict)
        for key, value in space.items:
            assert key.__module__ in [keras.layers, keras.activations]
            assert value is None or isinstance(value, tuple)
            for parameter in value:
                assert isinstance(parameter[0], str)
                for parameter_range in parameter[1:]:
                    assert isinstance(parameter_range, list) or isinstance(parameter_range, tuple)
        assert max_learnable_params == -1 or max_learnable_params > 0

        self.max_learnable_params = max_learnable_params
        self.space = space
        self.fitness = fitness

    def mutate(self, chromosome):
        pass

    def generate(self):
        pass


# immutable
class Allele:
    def __hash__(self):
        raise NotImplementedError(f'subclass does not implement {__name__}')

    def __eq__(self, other):
        raise NotImplementedError(f'subclass does not implement {__name__}')

    def can_crossover_with(self, other):
        return False

    def crossover(self, other):
        raise NotImplementedError()


class ParameterAllele(Allele):
    class Distribution:
        """Not equal by mere reference comparison. """

        @property
        def default(self):
            raise NotImplementedError(f'subclass does not implement {__name__}')

        def between(self, a, b):
            """ Returns a random element in this distribution between a and b (inclusive). """
            raise NotImplementedError(f'subclass does not implement {__name__}')

        def __contains__(self, item):
            raise NotImplementedError(f'subclass does not implement {__name__}')

        def __eq__(self, other):
            raise NotImplementedError(f'subclass does not implement {__name__}')

    class CollectionDistributionBase(Distribution):
        def __init__(self, collection, default):
            assert default is not None

            self._collection = collection
            self.__default = default

        def default(self):
            return self.__default

        def __contains__(self, item):
            return item in self._collection

        def __eq__(self, other):
            return self is other or (isinstance(other, __class__)
                                     and self._collection == other.__collection
                                     and self.__default == other.__default)

        def between(self, a, b):
            super().between(a, b)

    class CollectionDistribution(CollectionDistributionBase):
        def __init__(self, collection, default=None):
            assert len(collection) > 0

            default = default if default is not None else collection[len(collection) // 2]
            super().__init__(collection, default)

        def between(self, a, b):
            assert a in self
            assert b in self

            index_a = self._collection.index(a)
            index_b = self._collection.index(b)

            index_result = random.randint(min(index_a, index_b), max(index_a, index_b))
            return self._collection[index_result]

    class SetDistribution(CollectionDistributionBase):
        def __init__(self, *args, default=None):
            assert len(args) > 0

            default = default if default is not None else next(iter(args))
            super().__init__(args, default)

        def between(self, a, b):
            assert a in self
            assert b in self

            return random.choice(self.__collection)

    class DistributionValue(tuple):
        """A named tuple (parameter value, parameter distribution) """

        # noinspection PyInitNewSignature,PyArgumentList
        def __new__(cls, value, distribution):
            assert isinstance(distribution, ParameterAllele.Distribution)
            assert value in distribution

            return super(ParameterAllele.DistributionValue, cls).__new__(cls, (value, distribution))

        @property
        def value(self):
            return self[0]

        @property
        def distribution(self):
            return self[1]

        def __eq__(self, other):
            return isinstance(other, __class__) \
                   and self.value == other.value \
                   and self.distribution == other.distribution

        _hashes = {}
        def __hash__(self):
            try:
                return hash(self.value)
            except:
                try:
                    return __class__._hashes[(self.value,)]
                except KeyError:
                    __class__._hashes[(self.value,)] = len(__class__._hashes) + 1
                    return len(__class__._hashes)

    def __hash__(self):
        return self.__hash

    def _compute_hash(self):
        return sum(hash(name) * hash(value) for name, value in self.parameters.items())

    def __eq__(self, other):
        return self is other or (isinstance(other, __class__)
                                 and self.layer_type is other.layer_type
                                 and self.parameters == other.parameters)

    def __init__(self, layer_type, **parameters: Union[DistributionValue, Tuple[object, Distribution]]):
        super().__init__()
        # TODO: assert that layer_type is callable with the specified parameters

        # convert Tuple[object, Distribution] to DistributionValue:
        for key, value_and_distribution in parameters.items():
            if not isinstance(value_and_distribution, __class__.DistributionValue):
                assert isinstance(value_and_distribution, tuple) and len(value_and_distribution) == 2
                parameters[key] = __class__.DistributionValue(value_and_distribution[0], value_and_distribution[1])

        for key, (value, distribution) in parameters.items():
            if value is None:
                parameters[key] = distribution.default

        self.layer_type = layer_type
        self.parameters = parameters
        self.__hash = self._compute_hash()

    def can_crossover_with(self, other):
        return isinstance(other, ParameterAllele) and self.layer_type == other.layer_type

    def crossover(self, other: 'ParameterAllele'):
        assert self.can_crossover_with(other)

        # randomly select half of the unshared parameters
        nonoverlap = set(self.parameters.keys()).symmetric_difference(other.parameters.keys())
        result = {key: self.parameters[key] for key in nonoverlap if random.randint(0, 1) == 0}

        # select all parameters that are equal, and those that are unequal, choose something in between (inclusive)
        overlap = self.parameters.keys() & other.parameters.keys()
        for key in overlap:
            distribution = self.parameters[key].distribution
            value = distribution.between(self.parameters[key].value, other.parameters[key].value)
            result[key] = value, distribution
        return ParameterAllele(self.layer_type, **result)


class HyperChromosome:
    """Equal iff reference equals. """
    all_hc = {}

    def __init__(self, alleles: List[Allele], gene: Genome):
        super().__init__()
        self.__alleles = alleles
        self.gene = gene

        assert self not in __class__.all_hc

    @classmethod
    def create(cls, alleles: List[Allele]):
        result = HyperChromosome(alleles)
        if result in cls.all_hc:
            result = cls.all_hc[result]
        else:
            cls.all_hc[result] = result
        return result

    def clone(self):
        return HyperChromosome([allele for allele in self.__alleles])

    @staticmethod
    def crossover(a, b):
        assert isinstance(a, HyperChromosome)
        assert isinstance(b, HyperChromosome)
        assert a != b

        def alleles_equal(alleles: Tuple[Allele, Allele]):
            return alleles[0] == alleles[1]

        head = list(allele for allele, _ in itertools.takewhile(alleles_equal, zip(a, b)))
        tail = list(allele for allele, _ in itertools.takewhile(alleles_equal, zip(a[-1::], b[-1::])))

        remaining = ((a[i], b[i]) for i in range(len(head), min(len(a), len(b)) - len(tail)))
        for allele_a, allele_b in remaining:
            if allele_a.can_crossover_with(allele_b):
                head.append(allele_a.crossover_with(allele_b))

        for allele_a, allele_b in reversed(list(remaining)):
            if allele_a.can_crossover_with(allele_b):
                tail.append(allele_a.crossover_with(allele_b))

        tail_start = -len(tail) if len(tail) != 0 else None
        remaining_a = a[len(head):tail_start]
        remaining_b = b[len(head):tail_start]

        middle = list(__class__._randomly_mix(remaining_a, remaining_b))

        return __class__.create(head + middle + tail)

    @staticmethod
    def _randomly_mix(a: list, b: list):
        ai, bi = 0, 0

        while ai < len(a) or bi < len(b):
            if random.randint(0, len(a) + len(b)) > len(a):
                ai, bi = bi, ai
                a, b = b, a

            if ai < len(a):
                yield a[ai]
                ai += 1
            if random.uniform(0, 1) < len(b) / len(a):
                bi += 1

    def __hash__(self):
        return sum(hash(allele) for allele in self.__alleles)

    def __eq__(self, other):
        return self is other

    def __iter__(self):
        return self.__alleles


def ga(population_size, fitness, genome: Genome, *callbacks):
    from ga import ga
    ga(population_size,
       fitness,
       genome.generate,
       genome.mutate,
       HyperChromosome.crossover,
       HyperChromosome.clone,
       callbacks)


if __name__ == '__main__':
    for _ in range(200):
        x = list(HyperChromosome._randomly_mix([0, 1, 2], [3, 4, 5, 6, 7, 8, 9]))
        print(x)
