from hyperchrom import *
import keras
from keras.layers import *
from keras.activations import *
from itertools import *
from distribution import *
from random_dijkstra import SemiRandomDijkstraSavingAllRoutes, all_slotwise_combinations
import operator
import inspect
from functools import reduce
from integer_interval_union import IntegerInterval
from chromokeras_distributions import ReshapeDistributionFamily


class Node:
    def __init__(self, depth: int, shape: tuple, builder: Optional['ChromokerasAlleleBuilder'],
                 builder_parameters: Optional[Dict]):
        self.depth = depth
        self.shape = shape
        self.builder = builder
        self.builder_parameters = builder_parameters

    @property
    def is_real_layer(self):
        return self.builder.is_real_layer if self.builder else True

    def __getitem__(self, item):
        if item == 0:
            return self.depth
        elif item == 1:
            return self.shape
        elif item == 2:
            return self.builder
        else:
            raise IndexError()

    def __hash__(self):
        return hash((self.depth, self.shape, self.is_real_layer))

    def __eq__(self, other: 'Node'):
        # builder is not taken into account: it's the value; not the key
        return self.depth == other.depth and self.shape == other.shape # and self.is_real_layer == other.is_real_layer

    def __repr__(self):
        return f'node(depth={self.depth}, shape={self.shape}, {"None" if self.builder is None else self.builder.layer_type.__name__})'


_cached_nns = {}


def sequence_equals_with_ones_in_between(seq1, seq2):
    """ Returns whether the specified sequences are equal, except that one may have extra 1's in the sequence. """

    return all(item1 == item2 for item1, item2 in zip(filter(lambda i: i != 1, seq1), filter(lambda i: i != 1, seq2)))


def leaky_relu(alpha):
    assert alpha >= 0
    return lambda x: relu(x, alpha)


def _create_layer(allele: ParameterAllele):
    return allele.layer_type(**allele.parameters)


def genome_to_nn(genome: Genome):
    return map(chromosome_to_nn, genome.chromosomes)


def chromosome_to_nn(chromosome: Chromosome):
    layers = []
    for allele in chromosome.alleles:
        assert isinstance(allele, ParameterAllele)
        try:
            nn_list = _cached_nns[allele]
            layer = next(nn for nn in nn_list if nn not in layers)
        except KeyError:
            layer = _create_layer(allele)
            _cached_nns[allele] = [layer]
        except StopIteration:
            layer = _create_layer(allele)
            _cached_nns[allele].append(layer)
        layers.append(layer)
    return layers


class ChromokerasAlleleBuilder(ParameterAlleleBuilder):
    default_distributions: dict = NotImplementedError("subclass must implement 'get_default_distributions'")
    # input_rank is a number, a list of numbers or a tuple of numbers signifying an inclusive range
    input_rank: IntegerInterval = IntegerInterval.empty
    # indicates whether this is an actual layer with nodes, as opposed to the implementation detail type of layer
    # used by keras layer that isn't really a layer, e.g. Reshape and Flatten
    is_real_layer = True

    def get_shape_influencing_parameter_names(self):
        """
        Gets the parameters on the ParameterAlleles that influence the output shape.
        These are defined by the parameters of the  method 'output_shape', excluding 'self' and 'input_shape'
        :return:
        """
        args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, _ = inspect.getfullargspec(self.output_shape)

        assert not varargs
        assert not varkw
        assert not defaults
        assert not kwonlyargs
        assert not kwonlydefaults

        assert len(args) >= 1
        assert args[0] == 'input_shape'

        del args[0:1]
        return args

    @staticmethod
    def output_shape(input_shape):
        """
        If None is returned, it means the layer cannot be applied
        """
        return input_shape

    def contains_input_rank(self, value):
        assert isinstance(self.input_rank, IntegerInterval)
        return value in self.input_rank

    def __init__(self, layer_type, **overriding_distributions):
        """
        :param overriding_distributions: Overrides the default parameters
        """
        super().__init__(layer_type, **overriding_distributions)


class DenseBuilder(ChromokerasAlleleBuilder):
    default_distributions = {'units': SetDistribution([10, 20, 50, 100, 200, 500, 1000], default=50)}
    input_rank = IntegerInterval(1)

    # noinspection PyMethodOverriding
    @staticmethod
    def output_shape(input_shape, units):
        return input_shape[:-1] + (units,)

    def __init__(self, input_rank=IntegerInterval(1), **distributions):
        """
        :param input_rank: Indicates the ranks of the layers after which a dense layer may occur. By default only 1.
        """
        self.input_rank = input_rank
        super().__init__(Dense, **distributions)


class Conv2DBuilder(ChromokerasAlleleBuilder):
    default_distributions = {'filters': SetDistribution([8, 16, 32, 64], default=32),
                             'kernel_size': SetDistribution([(2, 2), (3, 3)], default=(2, 2)),
                             'activation': SetDistribution([relu, sigmoid, tanh, linear,
                                                            leaky_relu(0.01), leaky_relu(0.1)], default=relu)
                             }
    input_rank = IntegerInterval(3)

    @classmethod
    def contains_input_rank(cls, value):
        # noinspection PyArgumentList
        return super().contains_input_rank(cls, value)

    # noinspection PyMethodOverriding
    @staticmethod
    def output_shape(input_shape, filters, kernel_size):
        """
        If None is returned, it means the layer cannot be applied
        """
        assert len(input_shape) == 3
        assert isinstance(filters, int)
        assert isinstance(kernel_size, int) or (isinstance(kernel_size, tuple) and len(kernel_size) == 2
                                                and isinstance(kernel_size[0], int) and isinstance(kernel_size[1], int))

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if input_shape[0] < kernel_size[0] or input_shape[0] < kernel_size[1]:
            return None  # output shape would contain dimension with size 0, which is not possible

        return input_shape[0] - kernel_size[0] + 1, input_shape[1] - kernel_size[1] + 1, filters

    def __init__(self, **distributions):
        super().__init__(Conv2D, **distributions)


class FlattenBuilder(ChromokerasAlleleBuilder):
    default_distributions = {}
    input_rank = IntegerInterval((1, 10000))
    is_real_layer = False

    @staticmethod
    def output_shape(input_shape):
        """
        If None is returned, it means the layer cannot be applied
        """
        if len(input_shape) == 1:
            return None  # input is already flat
        return reduce(operator.mul, input_shape),

    def __init__(self):
        super().__init__(Flatten)


class ReshapeBuilder(ChromokerasAlleleBuilder):
    default_distributions = {}
    input_rank = IntegerInterval((1, 10000))
    is_real_layer = False

    # noinspection PyMethodOverriding
    @staticmethod
    def output_shape(input_shape, target_shape):
        """
        If None is returned, it means the layer cannot be applied
        """
        if reduce(operator.mul, input_shape) != reduce(operator.mul, target_shape):
            return None  # input is not commensurate with target
        if input_shape == target_shape:
            return None  # this reshape would be the identity function
        if sequence_equals_with_ones_in_between(input_shape, target_shape):
            return None  # this would still be the identity function (in that ℕ² ⊂ ℕ³)
        return target_shape

    def __init__(self, ranks: Union[IntegerInterval, int, list, tuple], rank_derivative_sign=None):
        super().__init__(Reshape)
        self.family = None
        self.ranks = ranks if isinstance(ranks, IntegerInterval) else IntegerInterval(ranks)
        self.rank_derivative_sign = rank_derivative_sign
        self.family = ReshapeDistributionFamily(self.ranks, self.rank_derivative_sign)
        self.distributions['target_shape'] = self.family

    def create(self, target_shape, **parameters):
        return super().create(target_shape=target_shape, **parameters)

    def get_cumulative_mutation_weight(self, allele: ParameterAllele):
        return self.family.get_weight(allele.parameters['target_shape'])


class ChromokerasBuilder(ChromosomeBuilder):
    @staticmethod
    def _get_default_allele_builders(input_shape, output_shape):
        ranks_interval = IntegerInterval((len(input_shape), len(output_shape)))
        return [ReshapeBuilder(ranks_interval),
                DenseBuilder(ranks_interval),
                FlattenBuilder(),
                Conv2DBuilder()]

    def __init__(self, input_shape, output_shape, allele_builders: Iterable[ChromokerasAlleleBuilder]=None):
        super().__init__(allele_builders or __class__._get_default_allele_builders(input_shape, output_shape))
        assert isinstance(input_shape, tuple)
        assert len(input_shape) > 0
        assert all(input_shape), 'The input_shape dimensions cannot be None'
        assert isinstance(output_shape, tuple)
        assert len(output_shape) > 0
        assert all(output_shape), 'The output_shape dimensions cannot be None'

        self.batch_input_shape = (None,) + input_shape
        self.batch_output_shape = (None,) + output_shape

    def generate(self, **kwargs):
        assert all(kwarg in ['layer_count', 'rank_derivative_sign'] for kwarg in kwargs)

        try:
            layer_count = kwargs['layer_count']
            assert isinstance(layer_count, int) and layer_count >= 2, 'There must be at least 2 layers'
            del kwargs['layer_count']
        except KeyError:
            layer_count = 2

        result = []
        shape_determining_nodes = next(iter(self.find_random_routes(layer_count)))
        for node in shape_determining_nodes:
            builder = node.builder
            if builder is not None:
                result.append(builder.generate(**node.builder_parameters))

        return Chromosome.create(result)

    def find_random_routes(self, end: Union[Node, int], start: Node=None) -> Iterable[Tuple[Node]]:
        """
        Returns random routes from start nodes to destination nodes, ad infinitum.
        :param start: Optionally the start from which the routes start.
        :param end: The node at which the random route ends, or the depth of the layer at which all routes end
        """
        assert isinstance(end, (Node, int))
        assert isinstance(start, Node) or start is None
        start = start or Node(0, self.batch_input_shape[1:], None, None)
        if isinstance(end, int):
            end = Node(end, self.batch_output_shape[1:], None, None)

        def get_all_layers_that_start_on(node: Node):
            if node.depth > end.depth or (node.depth == end.depth and not node.is_real_layer):
                return
            if node.is_real_layer:
                applicable_builders = self.allele_builders
            else:
                applicable_builders = filter(lambda b: b.is_real_layer, self.allele_builders)

            for builder in applicable_builders:
                if builder.contains_input_rank(len(node.shape)):
                    relevant_parameters = builder.get_shape_influencing_parameter_names()
                    distributions = [builder.distributions[name].get_collection(node.shape) for name in
                                     relevant_parameters]
                    for parameter_combination in all_slotwise_combinations(distributions):
                        parameter_combination_dict = dict(zip(relevant_parameters, parameter_combination))
                        output_shape = builder.output_shape(node.shape, **parameter_combination_dict)
                        if output_shape is not None:
                            assert isinstance(output_shape, tuple)
                            new_depth = node.depth + builder.is_real_layer
                            yield Node(new_depth, output_shape, builder, parameter_combination_dict )

        # find all real layers first such that the function get_all_layers can choose the
        # builder of real or both real and fake layers
        def get_comparable(node):
            return -node.depth - (10000 if node.is_real_layer else 0)

        dijkstra = SemiRandomDijkstraSavingAllRoutes([start],
                                                     get_neighbors=get_all_layers_that_start_on,
                                                     f_is_dest=lambda node: node.depth == end.depth
                                                                            and node.shape == end.shape,
                                                     get_comparable=get_comparable)

        return dijkstra.find_random_routes()

    def mutate_large(self, chromosome: Chromosome):
        pass

    def set_input_shape(self, input_layer):
        input_layer.batch_input_shape = self.batch_input_shape

    def has_input_shape(self, input_layer):
        return input_layer.batch_input_shape == self.batch_input_shape
