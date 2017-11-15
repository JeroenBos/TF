from typing import *
import unittest


class ImmutableCache:
    def __init__(self, type_):
        assert isinstance(type_, type)

        self._all = {}
        self.type_ = type_

    def create_key(self, *args, **kwargs):
        return self._Key(self.compute_hash, *args, **kwargs)

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

    @staticmethod
    def compute_hash(*args, **kwargs):
        result = sum(hash(arg) if arg is not None else 0 for arg in args) \
             + sum(hash(key) * (hash(value) if value is not None else -5531) for key, value in kwargs.items())
        assert isinstance(result, int)
        return result

    class _Key:
        def __init__(self, compute_hash, *args, **kwargs):
            self.__args = args
            self.__kwargs = kwargs
            self.__hash = compute_hash(*args, **kwargs)
            assert isinstance(self.__hash, int)

        def __hash__(self):
            return self.__hash

        def __eq__(self, other):
            # to override this method, the subclass ImmutableClass could create its own _Key class
            # but I don't think it's going to be necessary any time soon
            return self.__args == other.__args and self.__kwargs == other.__kwargs

        @property
        def hash(self):
            return self.__hash


class ImmutableCacheList(ImmutableCache):
    def __init__(self, type_):
        super().__init__(type_)

    def create_key(self, elements):
        return super().create_key(*elements)


class ImmutableCacheParameterAllele(ImmutableCache): #TODO: Make more generic

    def __init__(self, type_):
        super().__init__(type_)

    def create_key(self, layer_type, **parameters):
        return super().create_key(layer_type, **parameters)

    def create(self, layer_type, **parameters):
        return super().create(layer_type, **parameters)


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
