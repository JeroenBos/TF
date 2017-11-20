from hyperchrom import *
import keras
from keras.layers import *
from keras.activations import *
from itertools import *
from distribution import *
from random_dijkstra import SemiRandomDijkstraSavingAllRoutes, all_slotwise_combinations
import collections
import inspect


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
        try:
            builder_str = ", " + str(self.builder.layer_type.__name__)
        except:
            builder_str = ""
        return f'node(depth={self.depth}, shape={self.shape}{builder_str})'


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
    input_rank = []

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
        return input_shape

    def contains_input_rank(self, value):
        assert isinstance(self.input_rank, (int, list, tuple))
        assert isinstance(self.input_rank, int) or all(isinstance(rank, int) for rank in self.input_rank)
        assert isinstance(self.input_rank, (int, list)) or (len(self.input_rank) == 2 and self.input_rank[0] <= self.input_rank[1])

        if isinstance(self.input_rank, int):
            return self.input_rank == value
        if isinstance(self.input_rank, list):
            return value in self.input_rank

        return self.input_rank[0] <= value <= self.input_rank[1]

    def __init__(self, layer_type, **distributions):
        """
        :param parameters: Overrides the default parameters
        """
        distributions = self.default_distributions if len(distributions) != 0 else dict(self.default_distributions,
                                                                                        **distributions)
        super().__init__(layer_type, **distributions)


class DenseBuilder(ChromokerasAlleleBuilder):
    default_distributions = {'units': SetDistribution([10, 20, 50, 100, 200, 500, 1000], default=50)}
    rank = 1

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
    rank = 3

    # noinspection PyMethodOverriding
    @staticmethod
    def output_shape(input_shape, filters, kernel_size):
        assert len(input_shape) == 3
        assert isinstance(filters, int)
        assert isinstance(kernel_size, int) or (isinstance(kernel_size, tuple) and len(kernel_size) == 2
                                                and isinstance(kernel_size[0], int) and isinstance(kernel_size[1], int))

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        return input_shape[0] - kernel_size[0] + 1, input_shape[1] - kernel_size[1] + 1, filters

    def __init__(self, **distributions):
        super().__init__(Conv2D, **distributions)


class FlattenBuilder(ChromokerasAlleleBuilder):
    default_distributions = {}
    input_rank = (1, 10000)

    @staticmethod
    def output_shape(input_shape):
        """
        If None is returned, it means the layer cannot be applied
        """
        return 1

    def __init__(self):
        super().__init__(Flatten)


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

        # if 'rank_derivative_sign' in kwargs:
        #     assert kwargs['rank_derivative_sign'] in [-1, 0, 1, None]
        #     if kwargs['rank_derivative_sign']:
        #         op = [operator.lt, operator.eq, operator.gt][kwargs['rank_derivative_sign'] + 1]
        #         assert op(len(self.batch_output_shape), len(self.batch_input_shape))

        result = []
        while len(result) != layer_count:
            input_rank = self.get_rank(result[-1]) if len(result) != 0 else len(self.batch_input_shape) - 1

            potential_builders = [builder for builder in self.allele_builders if builder.contains_input_rank(input_rank)]
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
            if node.depth == end.depth:
                return
            for builder in self.allele_builders:
                relevant_parameters = builder.get_shape_influencing_parameter_names()
                distributions = [builder.distributions[name].collection for name in relevant_parameters]
                for parameter_combination in all_slotwise_combinations(distributions):
                    output_shape = builder.output_shape(node.shape, **dict(zip(relevant_parameters, parameter_combination)))
                    if output_shape is not None:
                        yield Node(node.depth + 1, output_shape, builder.layer_type.__name__)

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
