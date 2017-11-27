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
    def __init__(self, depth, shape, builder):
        self.depth = depth
        self.shape = shape
        self.builder = builder

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
        return hash((self.depth, self.shape))

    def __eq__(self, other: 'Node'):
        # builder is not taken into account: it's the value; not the key
        return self.depth == other.depth and self.shape == other.shape

    def __repr__(self):
        return f'node(depth={self.depth}, shape={self.shape}, {self.builder})'


_cached_nns = {}


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

    @classmethod
    def contains_input_rank(cls, value):
        assert isinstance(cls.input_rank, IntegerInterval)
        return value in cls.input_rank

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

    def __init__(self, **distributions):
        super().__init__(Dense, **distributions)


class Conv2DBuilder(ChromokerasAlleleBuilder):
    default_distributions = {'filters': SetDistribution([8, 16, 32, 64], default=32),
                             'kernel_size': SetDistribution([(2, 2), (3, 3)], default=(2, 2)),
                             'activation': SetDistribution([relu, sigmoid, tanh, linear,
                                                            leaky_relu(0.01), leaky_relu(0.1)], default=relu)
                             }
    input_rank = IntegerInterval(3)

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
        return target_shape

    def __init__(self, ranks: Union[IntegerInterval, int, list, tuple], final_shapes, rank_derivative_sign=None):
        super().__init__(Reshape)
        assert isinstance(final_shapes, tuple) and len(final_shapes) == 2
        self.family = None
        self.ranks = ranks if isinstance(ranks, IntegerInterval) else IntegerInterval(ranks)
        self.rank_derivative_sign = rank_derivative_sign
        self.family = ReshapeDistributionFamily(self.ranks, *final_shapes, self.rank_derivative_sign)
        self.distributions['target_shape'] = self.family

    def create(self, target_shape, **parameters):
        return super().create(target_shape=target_shape, **parameters)

    def get_cumulative_mutation_weight(self, allele: ParameterAllele):
        return self.family.get_weight(allele.parameters['target_shape'])


class ChromokerasBuilder(ChromosomeBuilder):
    def __init__(self, input_shape, output_shape, allele_builders: Iterable[ChromokerasAlleleBuilder]):
        super().__init__(allele_builders)
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

        layer_count = kwargs['layer_count'] if 'layer_count' in kwargs else 2
        assert isinstance(layer_count, int) and layer_count >= 2, 'There must be at least 2 layers'
        del kwargs['layer_count']

        result = []
        while len(result) != layer_count:
            input_rank = self.get_rank(result[-1]) if len(result) != 0 else len(self.batch_input_shape) - 1

            potential_builders = [builder for builder in self.allele_builders if
                                  builder.contains_input_rank(input_rank)]
            builder = random.choice(potential_builders)
            new_layer = builder.create_random()

            result.extend(self.generate_infix_layers(result[-1] if len(result) > 0 else None, new_layer, **kwargs))
            result.append(new_layer)
            result.extend(self.generate_postfix_layers(new_layer, **kwargs))

    def find_random_routes(self, end: Union[Node, int]):
        """
        Returns random routes from start nodes to destination nodes, ad infinitum.
        :param end: The node at which the random route ends, or the depth of the layer at which all routes end
        """
        assert isinstance(end, (Node, int))
        start = Node(0, self.batch_input_shape[1:], None)
        if isinstance(end, int):
            end = Node(end, self.batch_output_shape[1:], None)

        def get_all_layers_that_start_on(node: Node):
            if node.depth > end.depth:
                return
            elif node.depth == end.depth:
                applicable_builders = filter(lambda b: not b.is_real_layer, self.allele_builders)
            else:
                applicable_builders = self.allele_builders

            for builder in applicable_builders:
                if builder.contains_input_rank(len(node.shape)):
                    relevant_parameters = builder.get_shape_influencing_parameter_names()
                    distributions = [builder.distributions[name].get_collection(node.shape) for name in   # TODO: I can probably remove the parameter node.shape again....
                                     relevant_parameters]
                    for parameter_combination in all_slotwise_combinations(distributions):
                        output_shape = builder.output_shape(node.shape,
                                                            **dict(zip(relevant_parameters, parameter_combination)))
                        if output_shape is not None:
                            assert isinstance(output_shape, tuple)
                            yield Node(node.depth + builder.is_real_layer, output_shape, builder.layer_type.__name__)

        dijkstra = SemiRandomDijkstraSavingAllRoutes([start],
                                                     get_neighbors=get_all_layers_that_start_on,
                                                     f_is_dest=lambda node: node.depth == end.depth
                                                                            and node.shape == end.shape,
                                                     get_comparable=lambda node: -node.depth)
        return dijkstra.find_random_routes()

    def _determine_layer_output_rank(self, input_rank, layer_count):
        final_output_rank = len(self.batch_output_shape) - 1
        if input_rank > final_output_rank:
            if final_output_rank != 1:
                raise NotImplementedError()

            # flatten with a uniform probability per layer:
            if random.randint(0, layer_count) == 0:
                return 1
            else:
                return input_rank

        if input_rank < final_output_rank:
            pass

    @staticmethod
    def get_rank(layer):
        raise NotImplementedError()

    def create_flatten_layer(self):
        return

    def generate_postfix_layers(self, _layer_before, _rank_derivative_sign=None):
        """
        :return: an iterable of layers. May be empty.
        """
        return iter([])

    def mutate_large(self, chromosome: Chromosome):
        pass

    def set_input_shape(self, input_layer):
        input_layer.batch_input_shape = self.batch_input_shape

    def has_input_shape(self, input_layer):
        return input_layer.batch_input_shape == self.batch_input_shape
