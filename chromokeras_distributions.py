from typing import *
import prime_defactorization
from distribution import DistributionFamily, CollectionDistribution
from hyperchrom import product
from integer_interval_union import IntegerInterval


class ReshapeDistributionFamily(DistributionFamily):
    def __init__(self,
                 ranks: IntegerInterval,
                 rank_derivative_sign=None):
        # the family key member is the (input_size, output_ranks), the distribution members are the input_shape (=target_shape)
        super().__init__()
        assert isinstance(ranks, IntegerInterval)
        assert len(ranks) > 0
        assert rank_derivative_sign in [-1, 0, 1, None]
        # Maybe later I'll add None, meaning it can in fact be bidirectional

        self.ranks = ranks
        self.rank_derivative_sign = rank_derivative_sign

    @property
    def default(self):
        raise AttributeError('This does not exist')

    def _create_distribution(self, key):
        assert isinstance(key, tuple) and len(key) == 2  # the key is supposed to be the input_size
        input_size, ranks = key
        return ReshapeDistribution(self, input_size, ranks)

    def _get_key(self, input_shape):
        assert isinstance(input_shape, tuple)

        return product(input_shape), self._get_output_ranks(input_shape)

    def _get_output_ranks(self, input_rank_or_shape):
        """ Gets the collection of ranks that can be the direct output rank"""

        input_rank = input_rank_or_shape if isinstance(input_rank_or_shape, int) else len(input_rank_or_shape)

        if self.rank_derivative_sign == 1:
            allowed_ranks_mask = IntegerInterval((input_rank, 10000))
        elif self.rank_derivative_sign == 0:
            allowed_ranks_mask = IntegerInterval(input_rank)
        elif self.rank_derivative_sign == -1:
            allowed_ranks_mask = IntegerInterval((0, input_rank))
        else:
            allowed_ranks_mask = IntegerInterval((0, 10000))

        return self.ranks.intersection(allowed_ranks_mask)

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
        return f'ReshapeDistributionFamily(ranks={self.ranks}, derivative={self.rank_derivative_sign})'

    def get_collection(self, input_shape):
        allowed_ranks = self._get_output_ranks(input_shape)

        return self[(product(input_shape), allowed_ranks)]


class ReshapeDistribution(CollectionDistribution):
    @staticmethod
    def get_input_size(allele):
        return product(allele.parameters['target_shape'])

    def __init__(self, family: ReshapeDistributionFamily, input_size: int, ranks: IntegerInterval):
        self.family = family
        self.ranks = ranks
        self.input_size = input_size
        collection = list(self._compute_collection())
        super().__init__(collection)

    def _compute_collection(self):
        for rank in self.ranks:
            for shape in prime_defactorization.defactorize(self.input_size, rank):
                yield shape

    def __eq__(self, other):
        return isinstance(other, ReshapeDistribution) and self.ranks == other.ranks

    def __repr__(self):
        return f'ReshapeDistribution(size={self.input_size}, ranks={self.ranks})'
