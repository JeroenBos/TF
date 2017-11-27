from typing import *
import operator
from itertools import *
import random
import collections
import itertools
from functools import reduce

import prime_defactorization

from integer_interval_union import IntegerInterval


def product(numbers):
    return reduce(operator.mul, numbers)


def convert_to_rank_derivative_sign(final_input_shape, final_output_shape):
    if len(final_input_shape) > len(final_output_shape):
        return -1
    elif len(final_input_shape) < len(final_output_shape):
        return 1
    else:
        return 0


def are_shapes_compatible_with_rank_derivative(rank_derivative_sign, final_input_shape, final_output_shape):
    assert rank_derivative_sign in [-1, 0, 1, None]
    if not rank_derivative_sign:
        return True

    op = [operator.lt, operator.eq, operator.gt][rank_derivative_sign + 1]
    return op(len(final_output_shape), len(final_input_shape))


def nth_or_throw(iterable, n):
    i = 0
    for e in iterable:
        if i == n:
            return e
        i += 1
    raise IndexError


def nth(iterable, n, default=None):
    """Returns the nth item or a default value"""
    return next(islice(iterable, n, None), default)


class Distribution:
    """Not equal by mere reference comparison. """

    @property
    def default(self):
        raise NotImplementedError("subclass does not implement 'default'")

    def between(self, a, b):
        """ Returns a random element in this distribution between a and b (inclusive). """
        raise NotImplementedError("subclass does not implement 'between'")

    def mutate(self, a):
        """ Returns a value in the current distribution nearby (but not equal to) the specified element. """
        raise NotImplementedError("subclass does not implement 'mutate'")

    def __contains__(self, item):
        raise NotImplementedError("subclass does not implement '__contains__'")

    def __eq__(self, other):
        raise NotImplementedError("subclass does not implement '__eq__'")

    def random(self):
        """Draws a random element from this distribution"""
        raise NotImplementedError("subclass does not implement 'random'")

    def get_weight(self, allele):
        """Gets the weight of this distribution.
        :param allele: An allele representative for which the weight is asked. This matters for distribution families. """

        raise NotImplementedError()


class CollectionDistributionBase(Distribution):
    def __init__(self, collection, default):
        assert default in collection
        for element in collection:
            assert isinstance(element, collections.Hashable), type(element)
        assert len(collection) > 0

        self._collection = collection
        self.__default = default

    @property
    def default(self):
        return self.__default

    def __len__(self):
        return len(self._collection)

    def get_weight(self, _allele):
        return len(self)

    def __contains__(self, item):
        return item in self._collection

    # noinspection PyProtectedMember
    def __eq__(self, other):
        return self is other or (isinstance(other, __class__)
                                 and self._collection == other._collection
                                 and self.__default == other.__default)

    def mutate(self, a):
        assert a in self
        assert len(self) > 0

        assert all([])

        len_collection_without_a = sum(1 for e in self._collection if e != a)

        if len_collection_without_a == 0:
            return None
        chosen_index = random.randint(0, len_collection_without_a - 1)
        return nth((e for e in self._collection if e != a), chosen_index)


    def between(self, a, b):
        super().between(a, b)

    def random(self):
        return random.choice(self._collection)

    def get_collection(self, _input_shape=None):
        # if I eventually implement a collection that could not possibly return a collection, I have a problem
        # in the algorithm calling this method (get_all_layers_that_start_on)
        return self._collection  # TODO: return immutable wrapper


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


class DistributionFamily(Distribution):
    def __init__(self):
        self.__cache: Dict[object, Distribution] = {}  # the object is the family member key, which is ordinary a part of the allele

    def __getitem__(self, key):
        """
        :param key: The key of the family member to obtain.
        """
        try:
            return self.__cache[key]
        except KeyError:
            result = self._create_distribution(key)
            self.__cache[key] = result
            return result

    def __contains__(self, a):
        branch = self[self._get_key(a)]
        return a in branch

    def between(self, a, b):
        branch = self[self._get_key(a)]
        return branch.between(a, b)

    def mutate(self, element):
        """
        :param element: An element in any of the family distributions of which to return a mutated form.
        """

        branch = self[self._get_key(element)]
        return branch.mutate(element)

    def get_weight(self, element):
        """
        :param element: An element in any of the family distributions.
        """
        branch = self[self._get_key(element)]
        return branch.get_weight(element)

    def __eq__(self, other):
        if self is other:
            return True
        raise NotImplementedError()

    @property
    def default(self):
        raise NotImplementedError()

    def random(self):
        raise NotImplementedError()

    def _create_distribution(self, key):
        """ A function that creates a new family member, given the key of that member"""
        raise NotImplementedError()

    def _get_key(self, element):
        """
        Converts a distribution element (in any of the family branches) to the key that identifies which
        family branch the element is a member of.
        :param element: An element in any of the family distributions.
        """
        return element


