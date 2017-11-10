from typing import *
import itertools
import random
import collections
import inspect


def assert_is_callable_with(f, parameter_names):
    args, _varargs, varkw, defaults, kwonlyargs, kwonlydefaults, _annotations = inspect.getfullargspec(f)

    if varkw:
        return

    for parameter_name in parameter_names:
        if parameter_name not in args and parameter_name not in kwonlyargs:
            raise AttributeError(f'unused parameter {parameter_name} provided')

    args_without_default = args[:(-len(defaults) if defaults else None)]
    kwonlyargs_without_default = list(set(kwonlyargs) - set(kwonlydefaults.keys()))

    mandatory_parameter_names = args_without_default + kwonlyargs_without_default
    for mandatory_parameter_name in mandatory_parameter_names:
        if mandatory_parameter_name not in parameter_names:
            raise AttributeError(f'{mandatory_parameter_name} must be provided')


def product(iterable):
    result = 1
    for element in iterable:
        result *= element
    return result


class Genome:
    def __init__(self, space, max_learnable_params=-1):
        assert isinstance(space, dict)
        for key, value in space.items:
            assert value is None or isinstance(value, tuple)
            for parameter in value:
                assert isinstance(parameter[0], str)
                for parameter_range in parameter[1:]:
                    assert isinstance(parameter_range, list) or isinstance(parameter_range, tuple)
        assert max_learnable_params == -1 or max_learnable_params > 0

        self.max_learnable_params = max_learnable_params
        self.space = space

    def mutate(self, chromosome):
        pass

    def generate(self):
        pass


# immutable
class Allele:
    def __hash__(self):
        raise NotImplementedError(f"subclass does not implement '__hash__'")

    def __eq__(self, other):
        raise NotImplementedError(f"subclass does not implement '__eq__'")

    def can_crossover_with(self, other):
        return False

    def crossover(self, other):
        raise NotImplementedError(f"subclass does not implement 'crossover'")

    def __init__(self, cumulative_mutation_count):
        assert cumulative_mutation_count >= 0

        self.cumulative_mutation_count = cumulative_mutation_count


class ParameterAllele(Allele):
    class Distribution:
        """Not equal by mere reference comparison. """

        @property
        def default(self):
            raise NotImplementedError(f"subclass does not implement 'default'")

        def between(self, a, b):
            """ Returns a random element in this distribution between a and b (inclusive). """
            raise NotImplementedError(f"subclass does not implement 'between'")

        @property
        def size(self):
            raise NotImplementedError(f"subclass does not implement 'size'")

        def __contains__(self, item):
            raise NotImplementedError(f"subclass does not implement '__contains__'")

        def __eq__(self, other):
            raise NotImplementedError(f"subclass does not implement '__eq__'")

    class CollectionDistributionBase(Distribution):
        def __init__(self, collection, default):
            assert default is not None
            for element in collection:
                assert isinstance(element, collections.Hashable)

            self._collection = collection
            self.__default = default

        def default(self):
            return self.__default

        @property
        def size(self):
            return len(self._collection)

        def __contains__(self, item):
            return item in self._collection

        # noinspection PyProtectedMember
        def __eq__(self, other):
            return self is other or (isinstance(other, __class__)
                                     and self._collection == other._collection
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
        def __init__(self, collection, default=None):
            assert len(collection) > 0

            default = default if default is not None else collection[0]
            super().__init__(collection, default)

        def between(self, a, b):
            assert a in self
            assert b in self

            return random.choice(self._collection)

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
        def distribution(self) -> 'ParameterAllele.Distribution':
            return self[1]

        def __eq__(self, other):
            return isinstance(other, __class__) \
                   and self.value == other.value \
                   and self.distribution == other.distribution

        def __hash__(self):
            return hash(self.value)

    def __hash__(self):
        return self.__hash

    def _compute_hash(self):
        return sum(hash(name) * hash(value) for name, value in self.parameters.items())

    def __eq__(self, other):
        return self is other or (isinstance(other, __class__)
                                 and self.layer_type is other.layer_type
                                 and self.parameters == other.parameters)

    def __init__(self, layer_type, **parameters: Union[DistributionValue, Tuple[object, Distribution]]):

        assert_is_callable_with(layer_type, parameters.keys())

        # convert Tuple[object, Distribution] to DistributionValue:
        for key, value_and_distribution in parameters.items():
            if not isinstance(value_and_distribution, __class__.DistributionValue):
                assert isinstance(value_and_distribution, tuple) and len(value_and_distribution) == 2
                parameters[key] = __class__.DistributionValue(value_and_distribution[0], value_and_distribution[1])

        for key, (value, distribution) in parameters.items():
            if value is None:
                parameters[key] = distribution.default

        self.layer_type = layer_type
        self.parameters: Dict[__class__.DistributionValue] = parameters
        self.__hash = self._compute_hash()

        cumulative_mutation_count = product(parameter.distribution.size for parameter in self.parameters.values())
        # the current allele does not count, even though you technically won't mutate to it
        cumulative_mutation_count -= 1

        super().__init__(cumulative_mutation_count )

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


class Chromosome:
    """Equal iff reference equals. """
    _all_hc = {}

    class _Key:
        """Makes the list type hashable. """

        def __init__(self, alleles):
            self.__alleles = alleles
            self.__hash = __class__._hash(alleles)

        def __hash__(self):
            return self.__hash

        def __eq__(self, other):
            return self.__alleles == other.__alleles

        @staticmethod
        def _hash(alleles):
            return sum(hash(allele) for allele in alleles)

    def __init__(self, alleles: List[Allele]):
        super().__init__()
        self.__alleles = alleles
        self._cumulative_mutation_count = sum(allele.cumulative_mutation_count for allele in self.__alleles)

        assert self._Key(alleles) not in __class__._all_hc

    @classmethod
    def create(cls, alleles: List[Allele]):
        # key = hashable representation of alleles
        # noinspection PyProtectedMember
        key = __class__._Key(alleles)

        if key in cls._all_hc:
            result = cls._all_hc[key]
        else:
            result = Chromosome(alleles)
            cls._all_hc[key] = result
        return result

    def clone(self):
        return Chromosome([allele for allele in self.__alleles])

    @staticmethod
    def crossover(a, b):
        assert isinstance(a, Chromosome)
        assert isinstance(b, Chromosome)
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
       Chromosome.crossover,
       Chromosome.clone,
       callbacks)
