from typing import *
from itertools import *
import random
import inspect
from immutable_cache import ImmutableCacheList, ImmutableCacheParameterAllele
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


def _mutate(builders, elements, get_weight, perform_mutation, cumulative_weight=None):
    """
    :param builders: The builders of the elements to perform the mutation on
    :param builders: The elements to perform the mutation on
    :param get_weight: A function taking a builder and element returning the relative weight that is should be mutated
    :param perform_mutation: The function taking a builder and element  returning the mutated form
    :return: The list of new elements
    """

    assert all(builder for builder in builders)
    assert all(element for element in elements)
    element_type = type(next(iter(elements)))

    def get_element(chromosome_builder):
        return elements[builders.index(chromosome_builder)]

    # contains the builders of all chromosomes that didn't change, and the chromosome that did change
    builders_and_new_element = weighted_change(builders,
                                               lambda builder: get_weight(builder, get_element(builder)),
                                               lambda builder: perform_mutation(builder, get_element(builder)),
                                               cumulative_weight)
    new_elements = [x if isinstance(x, element_type) else get_element(x) for x in builders_and_new_element]
    return new_elements


def weighted_change(seq, f_weight, f_change, cumulative_weight=None):
    elem_index_to_substitute = weighted_choice(seq, f_weight, cumulative_weight)
    return with_element_at(seq, elem_index_to_substitute, f_change)


def with_element_at(seq, index, get_value):
    return (elem if i is not index else get_value(elem) for i, elem in enumerate(seq))


def weighted_choice(seq, f_weight, cumulative_weight=None):
    cumulative_weight = cumulative_weight if cumulative_weight else sum(f_weight(elem) for elem in seq)
    assert isinstance(cumulative_weight, int), f'type of cumulative_weight: {type(cumulative_weight)}'

    cumulative_i = random.randint(0, cumulative_weight)

    result = 0
    for elem in seq:
        cumulative_i -= f_weight(elem)
        if cumulative_i <= 0:
            return result
        result += 1
    assert False


def product(iterable):
    result = 1
    for element in iterable:
        result *= element
    return result


# immutable
class Allele:
    def __hash__(self):
        raise NotImplementedError("subclass does not implement '__hash__'")

    def __eq__(self, other):
        raise NotImplementedError("subclass does not implement '__eq__'")


class ParameterAllele(Allele):
    _all = None

    def __hash__(self):
        assert self.hash, f'A {__name__} was created from __init__ rather than create(..)'
        return self.hash

    def __eq__(self, other):
        return self is other or (isinstance(other, __class__)
                                 and self.layer_type is other.layer_type
                                 and self.parameters == other.parameters)

    def __init__(self, layer_type, **parameters):
        """
        :param layer_type: The callable creating the layer.
        :param parameters: Pairs of names of parameters to layer_type.__call__. None is allowed as value.
        """
        assert len(parameters) >= 0
        assert all(not isinstance(parameter, Distribution) for parameter in parameters.values())
        assert all(
            not isinstance(p, tuple) or all(not isinstance(t, Distribution) for t in p) for p in parameters.values())
        assert_is_callable_with(layer_type, parameters.keys())
        assert inspect.getouterframes(inspect.currentframe())[1].function == 'create',\
            "You're not allowed to call __init__ directly; call create(..)"

        self.builder = None  # set by create method
        self.hash = None  # set by create method
        self.layer_type = layer_type
        self.parameters = parameters

    @classmethod
    def create(cls, builder, layer_type, **parameters):
        result = cls._all.create(layer_type, **parameters)
        result.builder = builder
        result.hash = ImmutableCacheParameterAllele.compute_hash(layer_type, **parameters)
        return result

    def __repr__(self):
        return f'{self.layer_type.__name__}({", ".join(str(key) + "=" + str(value) for key, value in self.parameters.items())})'


ParameterAllele._all = ImmutableCacheParameterAllele(ParameterAllele)


class Chromosome:
    """Equal iff reference equals. """
    _all = None

    def __init__(self, alleles: List[Allele]):
        assert all(allele for allele in alleles)
        assert inspect.getouterframes(inspect.currentframe())[1].function == 'create',\
            "You're not allowed to call __init__ directly; call create(..)"
        super().__init__()
        self.alleles = alleles

    def clone(self):
        return self  # Chromosome is immutable so

    @classmethod
    def create(cls, alleles):
        return cls._all.create(alleles)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        # noinspection PyUnresolvedReferences
        return self.hash  # set in immutable_cache


Chromosome._all = ImmutableCacheList(Chromosome)


class Genome:
    _all = None

    def __init__(self, chromosomes):
        assert isinstance(chromosomes, list)
        assert inspect.getouterframes(inspect.currentframe())[1].function == 'create',\
            "You're not allowed to call __init__ directly; call create(..)"

        self.chromosomes = chromosomes

    @classmethod
    def create(cls, chromosomes):
        return cls._all.create(chromosomes)

    def clone(self):
        return self  # Genome is immutable so


Genome._all = ImmutableCacheList(Genome)


class AlleleBuilder:
    def __init__(self, cumulative_mutation_weight):
        """
        :param cumulative_mutation_weight: The number mutations an allele built by this builder can undergo.
        """
        self.cumulative_mutation_weight = cumulative_mutation_weight

    def mutate(self, allele):
        raise NotImplementedError("subclass must implement 'mutate'")


class ParameterAlleleBuilder(AlleleBuilder):
    def __init__(self, layer_type, **distributions):
        assert len(distributions) >= 0
        assert all(isinstance(distribution, Distribution) for distribution in distributions.values())
        assert_is_callable_with(layer_type, distributions.keys())

        self.layer_type = layer_type
        self.distributions = distributions

        cumulative_mutation_weight = product(len(distribution) for distribution in self.distributions.values())
        # the current allele does not count, even though you technically won't mutate to it
        cumulative_mutation_weight -= 1
        super().__init__(cumulative_mutation_weight)

    def mutate(self, allele):
        def get_weight(param_value_pair):
            parameter, value = param_value_pair
            distribution = self.distributions[parameter]
            return len(distribution)

        def mutate(param_value_pair):
            parameter, value = param_value_pair
            distribution = self.distributions[parameter]
            new_value = distribution.mutate(value)
            assert new_value != value
            return parameter, new_value

        new_parameters = dict(weighted_change(allele.parameters.items(),
                                              get_weight,
                                              mutate,
                                              self.cumulative_mutation_weight))

        return ParameterAllele.create(self, allele.layer_type, **new_parameters)

    def crossover(self, a: 'ParameterAllele', b: 'ParameterAllele'):
        assert self.can_crossover_with(a, b)
        assert all(key in self.distributions for key in chain(a.parameters.keys(), b.parameters.keys()))

        # randomly select half of the unshared parameters
        nonoverlap = set(a.parameters.keys()).symmetric_difference(b.parameters.keys())
        result = {key: a.parameters[key] for key in nonoverlap if random.randint(0, 1) == 0}

        # select all parameters that are equal, and those that are unequal, choose something in between (inclusive)
        overlap = a.parameters.keys() & b.parameters.keys()
        for key in overlap:
            distribution = self.distributions[key]
            result[key] = distribution.between(a.parameters[key], b.parameters[key])
        return ParameterAllele.create(self, self.layer_type, **result)

    def generate(self):
        """
        Draws a random element from this distribution.
        """
        parameters = {name: distribution.random() for name, distribution in self.distributions.items()}
        return ParameterAllele.create(self, self.layer_type, **parameters)

    def default(self):
        """
        Gets the allele with default parameters.
        """
        parameters = {name: distribution.default for name, distribution in self.distributions.items()}
        return ParameterAllele.create(self.layer_type, **parameters)

    def create(self, **parameters):
        assert all(parameter in self.distributions for parameter in parameters), \
            f'Unknown parameter specified: {next(p for p in parameters if parameters not in self.distributions)}'

        for name, distribution in self.distributions.items():
            if name not in parameters:
                parameters[name] = distribution.default

        return ParameterAllele.create(self, self.layer_type, **parameters)

    def create_random(self):
        parameters = {name: distribution.random() for name, distribution in self.distributions.items()}
        return ParameterAllele.create(self, self.layer_type, **parameters)

    @staticmethod
    def can_crossover_with(a: ParameterAllele, b: ParameterAllele):
        return isinstance(a, ParameterAllele) and isinstance(b, ParameterAllele) and a.layer_type == b.layer_type


class ChromosomeBuilder:
    """Defines the constraints imposed on a genome and its alleles and their occurrences and order, etc. """

    def __init__(self, allele_builders: Iterable[ParameterAlleleBuilder]):
        self.allele_builders = allele_builders

    def mutate_large(self, chromosome: Chromosome):
        """Mutates the shape of chromosomes, taking into account the constraints. """
        raise NotImplementedError("subclass must implement 'mutate_large'")

    def mutate_small(self, chromosome: Chromosome):
        return Chromosome.create(_mutate(self.allele_builders,
                                         chromosome.alleles,
                                         lambda builder, allele: builder.cumulative_mutation_weight,
                                         lambda builder, allele: builder.mutate(allele)))

    def generate(self):
        """Returns a random chromosome subject to the constraints imposed by this builder"""
        raise NotImplementedError("subclass must implement 'generate'")

    @staticmethod
    def can_mutate():
        """Returns whether this chromosome is not constrained to no change at all"""
        return True

    def get_large_mutation_weight(self, chromosome):
        if self.can_mutate:
            return len(chromosome.alleles)
        return 0

    # noinspection PyMethodMayBeStatic
    def get_small_mutation_weight(self, chromosome):
        return sum(allele.builder.cumulative_mutation_weight for allele in chromosome.alleles)

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

        return Chromosome.create(head + middle + tail)

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

    @staticmethod
    def create(alleles: List[Allele]):
        result: Chromosome = Chromosome.create(alleles)
        return result


class GenomeBuilder:
    """Defines the constraints imposed on a genome and its alleles and their occurrences and order, etc. """

    def __init__(self, *chromosome_builders, large_mutation_probability=0.2):
        assert len(chromosome_builders) > 0
        assert all(isinstance(cb, ChromosomeBuilder) for cb in chromosome_builders)

        self.chromosome_builders = chromosome_builders
        self.large_mutation_probability = large_mutation_probability

    @property
    def n(self):
        return len(self.chromosome_builders)

    def generate(self):
        """Returns a random genome subject to their constraints"""
        return Genome.create([builder.generate() for builder in self.chromosome_builders])

    def mutate(self, genome: Genome):
        """Does a small or large mutation and returns the result"""

        # this method is responsible for choosing whether a large or small mutation is done
        if random.uniform(0, 1) < self.large_mutation_probability:
            return self._mutate_large(genome)
        else:
            return self._mutate_small(genome)

    def _mutate_small(self, genome):
        """Does a small mutation and returns the new genome"""
        return Genome.create(_mutate(self.chromosome_builders,
                                     genome.chromosomes,
                                     get_weight=ChromosomeBuilder.get_small_mutation_weight,
                                     perform_mutation=ChromosomeBuilder.mutate_small,
                                     cumulative_weight=None))

    def _mutate_large(self, genome):
        """Does a small mutation and returns the new genome"""
        return Genome.create(_mutate(self.chromosome_builders,
                                     genome.chromosomes,
                                     get_weight=ChromosomeBuilder.get_large_mutation_weight,
                                     perform_mutation=ChromosomeBuilder.mutate_large,
                                     cumulative_weight=None))

    def crossover(self, a: 'Genome', b: 'Genome'):
        assert len(a.chromosomes) == len(b.chromosomes)

        def _implementation():
            for builder, c1, c2 in zip(self.chromosome_builders, a.chromosomes, b.chromosomes):
                yield builder.crossover(c1, c2)

        return Genome.create(list(_implementation()))


def ga(population_size, fitness, builder: Union[GenomeBuilder, ChromosomeBuilder], *callbacks):
    builder = builder if isinstance(builder, GenomeBuilder) else GenomeBuilder(builder)
    from ga import ga
    ga(population_size,
       fitness,
       builder.generate,
       builder.mutate,
       builder.crossover,
       Genome.clone,
       callbacks)
