from itertools import *
import random
import collections


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

    @property
    def size(self):
        raise NotImplementedError("subclass does not implement 'size'")

    def __contains__(self, item):
        raise NotImplementedError("subclass does not implement '__contains__'")

    def __eq__(self, other):
        raise NotImplementedError("subclass does not implement '__eq__'")

    def random(self):
        """Draws a random element from this distribution"""
        raise NotImplementedError("subclass does not implement 'random'")


class CollectionDistributionBase(Distribution):
    def __init__(self, collection, default):
        assert default in collection
        for element in collection:
            assert isinstance(element, collections.Hashable)

        self._collection = collection
        self.__default = default

    @property
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

    def mutate(self, a):
        assert a in self
        assert self.size != 1

        collection_without_a = filterfalse(lambda e: e is not a, self._collection)
        multiplicity_a = sum(1 for _ in collection_without_a)
        return nth(collection_without_a, random.randint(0, self.size - 1 - multiplicity_a))

    def between(self, a, b):
        super().between(a, b)

    def random(self):
        return random.choice(self._collection)

    @property
    def collection(self):
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
