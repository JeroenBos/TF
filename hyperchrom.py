from typing import *
from itertools import *
import random
import inspect
from immutable_cache import ImmutableCacheList
from distribution import Distribution


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


def weighted_change(seq, f_weight, f_change, cumulative_weight=None):
    elem_to_substitute = weighted_choice(seq, f_weight, cumulative_weight)
    return (elem if elem is not elem_to_substitute else f_change(elem_to_substitute) for elem in seq)


def weighted_choice(seq, f_weight, cumulative_weight=None):
    cumulative_weight = cumulative_weight if cumulative_weight else sum(f_weight(elem) for elem in seq)
    assert isinstance(cumulative_weight, int), f'type of cumulative_weight: {type(cumulative_weight)}'

    cumulative_i = random.randint(0, cumulative_weight)

    for elem in seq:
        cumulative_i -= f_weight(elem)
        if cumulative_i <= 0:
            return elem
    assert False


def product(iterable):
    result = 1
    for element in iterable:
        result *= element
    return result


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

    def mutate(self):
        raise NotImplementedError(f"subclass {self.__class__.__name__} does not implement 'mutate'")

    @property
    def cumulative_mutation_count(self):
        return self.__cumulative_mutation_count

    def get_cumulative_mutation_count(self):
        return self.__cumulative_mutation_count

    def __init__(self, cumulative_mutation_count):
        assert cumulative_mutation_count >= 0

        self.__cumulative_mutation_count = cumulative_mutation_count


class ParameterAllele(Allele):
    class DistributionValue(tuple):
        """A named tuple (parameter value, parameter distribution) """

        # noinspection PyInitNewSignature,PyArgumentList
        def __new__(cls, value, distribution):
            assert isinstance(distribution, Distribution)
            assert value in distribution

            return super(ParameterAllele.DistributionValue, cls).__new__(cls, (value, distribution))

        @property
        def value(self):
            return self[0]

        @property
        def distribution(self) -> 'Distribution':
            return self[1]

        def __eq__(self, other):
            return isinstance(other, __class__) \
                   and self.value == other.value \
                   and self.distribution == other.distribution

        def __hash__(self):
            return hash(self.value)

        def mutate(self):
            return self.distribution.mutate(self.value), self.distribution

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
        self.parameters: Dict[str, __class__.DistributionValue] = parameters
        self.__hash = self._compute_hash()

        cumulative_mutation_count = product(parameter.distribution.size for parameter in self.parameters.values())
        # the current allele does not count, even though you technically won't mutate to it
        cumulative_mutation_count -= 1

        super().__init__(cumulative_mutation_count)

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

    def mutate(self):
        assert self.cumulative_mutation_count != 0

        def get_weight(str_dis_val: Tuple[str, self.DistributionValue]):
            return str_dis_val[1].distribution.size

        new_parameters = weighted_change(self.parameters.items(),
                                         get_weight,
                                         self.DistributionValue.mutate,
                                         self.cumulative_mutation_count)
        new_parameters = dict(new_parameters)
        return __class__(self.layer_type, **new_parameters)


class Chromosome:
    """Equal iff reference equals. """
    _all = None

    def __init__(self, alleles: List[Allele]):
        super().__init__()
        self.__alleles = alleles
        self.__cumulative_mutation_count = sum(allele.cumulative_mutation_count for allele in self.__alleles)

        assert alleles not in self._all, \
            f'{self.__class__.__name__}s must be created through {self.__class__.__name__}.create(...)'

    def clone(self):
        return self  # Chromosome is immutable so

    @property
    def alleles(self):
        return self.__alleles

    @classmethod
    def create(cls, alleles):
        return cls._all.create(alleles)

    @staticmethod
    def crossover(a, b):
        assert isinstance(a, Chromosome)
        assert isinstance(b, Chromosome)
        assert a != b

        def alleles_equal(alleles: Tuple[Allele, Allele]):
            return alleles[0] == alleles[1]

        head = list(allele for allele, _ in takewhile(alleles_equal, zip(a, b)))
        tail = list(allele for allele, _ in takewhile(alleles_equal, zip(a[-1::], b[-1::])))

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

    def mutate(self):
        new_alleles = weighted_change(self.__alleles,
                                      Allele.get_cumulative_mutation_count,
                                      lambda allele: allele.mutate(),  # Allele.mutate doesn't call overridden method
                                      self.__cumulative_mutation_count)
        return self.create(new_alleles)

    @property
    def cumulative_mutation_count(self):
        return self.__cumulative_mutation_count

    def get_cumulative_mutation_count(self):
        return self.cumulative_mutation_count

    def __eq__(self, other):
        return self is other

    def __iter__(self):
        return self.__alleles

    def __hash__(self):
        # noinspection PyUnresolvedReferences
        return self.hash  # set in immutable_cache


Chromosome._all = ImmutableCacheList(Chromosome)


class Genome:
    _all = None

    def __init__(self, chromosomes):
        assert isinstance(chromosomes, list)

        self.__chromosomes = chromosomes
        self._cumulative_mutation_count = sum(chromosome.cumulative_mutation_count for chromosome in chromosomes)

        assert chromosomes not in self._all

    @classmethod
    def create(cls, chromosomes):
        return cls._all.create(chromosomes)

    @property
    def chromosomes(self):
        return self.__chromosomes

    def mutate_small(self):
        """Does a small mutation and returns the new genome"""
        new_chromosomes = weighted_change(self.chromosomes,
                                          Chromosome.get_cumulative_mutation_count,
                                          Chromosome.mutate,
                                          self._cumulative_mutation_count)
        return __class__.create(list(new_chromosomes))

    @staticmethod
    def crossover(self: 'Genome', other: 'Genome'):
        assert len(self.chromosomes) == len(other.chromosomes)

        return __class__.create(list(Chromosome.crossover(*pair) for pair in zip(self.chromosomes, other.chromosomes)))

    def clone(self):
        return self  # Genome is immutable so


Genome._all = ImmutableCacheList(Genome)


class ChromosomeBuilder:
    """Defines the constraints imposed on a genome and its alleles and their occurrences and order, etc. """
    def __init__(self, alleles):
        self.alleles = alleles

    def mutate_shape(self, chromosome: Chromosome):
        """Mutates the shape of chromosomes, taking into account the constraints. """
        raise NotImplementedError("subclass must implement 'mutate'")

    def generate(self):
        """Returns a random chromosome subject to the constraints imposed by this builder"""
        raise NotImplementedError("subclass must implement 'generate'")

    def can_mutate(self):
        """Returns whether this chromosome is not constrained to no change at all"""
        return True


class GenomeBuilder:
    """Defines the constraints imposed on a genome and its alleles and their occurrences and order, etc. """
    def __init__(self, *chromosome_builders, large_mutation_probability=0.2):
        assert len(chromosome_builders) > 0
        assert all(isinstance(cb, int) for cb in chromosome_builders)

        self.chromosome_builders = chromosome_builders
        self.large_mutation_probability = large_mutation_probability

    @property
    def n(self):
        return len(self.chromosome_builders)

    def generate(self):
        """Returns a random genome subject to their constraints"""
        return Genome.create([builder.generate() for builder in self.chromosome_builders])

    def mutate(self, genome: Genome):
        """Does a small or large mutation and return the result"""

        # this method is responsible for choosing whether a large or small mutation is done
        if random.uniform(0, 1) < self.large_mutation_probability:
            c, cb = weighted_choice(zip(genome.chromosomes, self.chromosome_builders), self._chromosome_large_mutation_weight)
            mutated_c = cb.mutate_shape()
            return Genome.create(list(c_ if c is not c_ else mutated_c for c_ in genome.chromosomes))
        else:
            return genome.mutate_small()

    @staticmethod
    def _chromosome_large_mutation_weight(chromosome, chromosome_builder):
        if chromosome_builder.can_mutate:
            return len(chromosome.alleles)
        return 0


def ga(population_size, fitness, builder: GenomeBuilder, *callbacks):
    builder = builder if isinstance(builder, GenomeBuilder) else GenomeBuilder(builder)
    from ga import ga
    ga(population_size,
       fitness,
       builder.generate,
       builder.mutate,
       Genome.crossover,
       Genome.clone,
       callbacks)
