from typing import *
import prime_defactorization
from distribution import DistributionFamily, convert_to_rank_derivative_sign, CollectionDistribution
from hyperchrom import product
from integer_interval_union import IntegerInterval


class ReshapeDistributionFamily(DistributionFamily):
    def __init__(self,
                 ranks: IntegerInterval,
                 final_input_shape: int,
                 final_output_shape: int,
                 rank_derivative_sign=None):
        # the family key member is the input_size, the distribution members ais the input_shape (=target_shape)
        super().__init__()
        assert isinstance(ranks, IntegerInterval)
        assert isinstance(final_input_shape, tuple)
        assert isinstance(final_output_shape, tuple)
        assert rank_derivative_sign in [-1, 0, -1, None]
        # Maybe later I'll add None, meaning it can in fact be bidirectional

        self.ranks = ranks
        self.final_input_shape = final_input_shape
        self.final_output_shape = final_output_shape
        self.rank_derivative_sign = rank_derivative_sign \
            if rank_derivative_sign else convert_to_rank_derivative_sign(final_input_shape, final_output_shape)

    def random(self):
        raise NotImplementedError()

    @property
    def default(self):
        raise AttributeError('This does not exist')

    def _create_distribution(self, key):
        assert isinstance(key, int)  # the key is supposed to be the input_size
        input_size = key
        return ReshapeDistribution(self, input_size)

    def _get_key(self, input_shape):
        assert isinstance(input_shape, tuple)

        return product(input_shape)

    def __contains__(self, element):
        """
        :param element: An element in any of the distributions (or at least, that is to be checked).
        """
        #  The method checks whether the element satisfies the final_input to final_output constraints
        assert isinstance(element, tuple)  # it's the input_shape

        def _complies_with_rank_derivatives(self, middle):
            def _complies_with_rank_derivative(first: tuple, second: tuple, derivative: int):
                assert derivative in [-1, 0, 1]  # none must already have been filtered out

            if self.rank_derivative_sign:
                return _complies_with_rank_derivative(self.final_input_shape, middle, self.rank_derivative_sign)

            raise NotImplementedError('fack it, I\'ll do it later')

        return len(element) in self.ranks and super().__contains__(element)

    def __repr__(self):
        return f'ReshapeDistributionFamily(ranks={self.ranks}, final_input={self.final_input_shape}, ' \
               f'final_output={self.final_output_shape}, derivative={self.rank_derivative_sign})'


class ReshapeDistribution(CollectionDistribution):
    @staticmethod
    def get_input_size(allele):
        return product(allele.parameters['target_shape'])

    def __init__(self, family: ReshapeDistributionFamily, input_size: int):
        self.family = family
        self.input_size = input_size
        collection = list(self._compute_collection())
        super().__init__(collection)

    @property
    def ranks(self):
        return self.family.ranks

    @property
    def final_input_shape(self):
        return self.family.final_input_shape

    @property
    def final_output_shape(self):
        return self.family.final_output_shape

    def _compute_collection(self):
        for output_rank in self.ranks:
            yield tuple(prime_defactorization.defactorize(self.input_size, output_rank))

    def __eq__(self, other):
        return isinstance(other, ReshapeDistribution) and self.ranks == other.ranks

    def __contains__(self, item):
        return isinstance(item, tuple) and len(item) in self.ranks

    def __repr__(self):
        return f'ReshapeDistribution(size={self.input_size}, ranks={self.ranks})'
