from typing import *
import unittest


class ImmutableCache:
    def __init__(self, type_):
        assert isinstance(type_, type)

        self._all = {}
        self.type_ = type_

    def create_key(self, *args, **kwargs):
        raise NotImplementedError("subclass must implement 'create_key'")

    def create(self, *args, **kwargs):
        key = self.create_key(*args, **kwargs)
        if key in self._all:
            return self._all[key]
        else:
            result = self.type_(*args, **kwargs)
            result.hash = key.hash
            self._all[key] = result
            return result

    def __contains__(self, *args):
        return self.create_key(*args) in self._all


class ImmutableCacheList(ImmutableCache):
    class _Key:
        """Makes the list type hashable. """

        def __init__(self, elements):
            self.__elements = elements
            self.__hash = __class__._hash(elements)

        def __hash__(self):
            return self.__hash

        def __eq__(self, other):
            return self.__elements == other.__elements

        @staticmethod
        def _hash(elements):
            return sum(hash(element) for element in elements)

        @property
        def hash(self):
            return self.__hash

    def __init__(self, type_):
        super().__init__(type_)

    def create_key(self, *args, **kwargs):
        return self._Key(*args, **kwargs)

    def create(self, elements: List):
        return super().create(elements)


class Tests(unittest.TestCase):
    class MockList:
        def __init__(self, _elements):
            pass

    def test_reference_equality(self):
        cache = ImmutableCacheList(Tests.MockList)
        a = cache.create([0, 1])
        b = cache.create([0, 1])

        self.assertIs(a, b)

    def test_reference_inequality(self):
        cache = ImmutableCacheList(Tests.MockList)
        a = cache.create([0, 1])
        b = cache.create([1, 2])

        self.assertIsNot(a, b)


if __name__ == '__main__':
    unittest.main()
