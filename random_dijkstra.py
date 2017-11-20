from typing import *
import heapq
import random
import sys
import collections
import itertools

T = TypeVar('T')
multiplicityNode = collections.namedtuple('multicityNode', ['value', 'multiplicity'])


class RandomWalk:
    """
        Gets a route by walking randomly from the root of a tree to any destination node.
    """

    def __init__(self,
                 start: Iterable[T],
                 get_neighbors: Callable[[T], Iterable],
                 f_is_dest: Callable[[T], bool]):
        assert all(isinstance(start_, collections.Hashable) for start_ in start)
        self.open = self.create_open((self._wrap(None, start_) for start_ in start))
        self._get_neighbors = get_neighbors
        self._is_dest = f_is_dest
        self.closed = {}

    def create_open(self, start):
        return {start_node: None for start_node in start}

    def pop(self):
        return self.open.popitem()

    def push(self, current, new):
        self.open[new] = current

    def add_edge_to_closed(self, current, new):
        self.closed[new] = current

    def update_closed(self, current, new):
        return new in self.closed

    def find_route(self):
        """
        Returns all independent routes from any destination node to any start node (so in reverse).
        :return:
        """
        assert len(self.closed) == 0, 'this method may only be called once'

        while len(self.open) != 0:
            node_from, current = self.pop()
            self.add_edge_to_closed(node_from, current)
            if self._is_dest(current):
                yield self._generate_result(current)

            for new in self._get_neighbors(current):
                assert new, 'Neighbor may not be None'
                was_already_in_closed = self.update_closed(current, new)
                if not was_already_in_closed:
                    self.push(current, new)

    def _generate_result(self, end):
        while end is not None:
            yield end
            end = self.get_from(end)

    def get_from(self, end):
        """
        :return: The first node from which can be walked to the specified node
        """
        return self.closed[end]

    def _wrap(self, node_from, current):
        return current


class RandomWalkSavingAllRoutes(RandomWalk):
    """
        Gets a route by walking randomly from the root of a tree to any destination node,
        and maintains all routes to that destination rather than an arbitrary one.
    """

    def push(self, current, new):
        self.open[new] = current

    def add_edge_to_closed(self, current, new):
        self.closed[new] = [current]

    def update_closed(self, current, new):
        if new in self.closed:
            self.closed[new].append(current)
            return True
        return False

    def get_from(self, end):
        """
        :return: The first node from which can be walked to the specified node
        """
        return self.closed[end][0]


class SemiRandomDijkstra(RandomWalk):
    """
    Implements a depth first random walk. I don't know why I called it dijkstra, because there are no costs.
    """

    class Node:
        def __init__(self, current, new, new_comparable):
            self.__current = current
            self.__new = new
            self.__new_comparable = new_comparable
            self.__randomization = random.randint(0, sys.maxsize)

        @property
        def current(self):
            return self.__current

        @property
        def new(self):
            return self.__new

        def __lt__(self, other):
            assert isinstance(other, __class__)
            return self.__new_comparable < other.__new_comparable \
                   or (self.__new_comparable == other.__new_comparable
                       and self.__randomization < other.__randomization)

        def __gt__(self, other):
            assert isinstance(other, __class__)
            return self.__new_comparable > other.__new_comparable \
                   or (self.__new_comparable == other.__new_comparable
                       and self.__randomization > other.__randomization)

        def __getitem__(self, item):
            if item == 0:
                return self.current
            elif item == 1:
                return self.new
            else:
                raise IndexError()

        def __repr__(self):
            return f'Node({self.current}->{self.new})'

    def __init__(self,
                 start,
                 get_neighbors,
                 f_is_dest,
                 get_comparable: Callable[[T], object]):
        self.get_comparable = get_comparable
        super().__init__(start, get_neighbors, f_is_dest)

    def create_open(self, start_wr):
        """
        :param start_wr: iterable of wrapped starting elements
        """
        heap = list(start_wr)
        heapq.heapify(heap)
        return heap

    def push(self, current, new):
        new_wr = self._wrap(current, new)
        heapq.heappush(self.open, new_wr)

    def pop(self):
        """
        :return: a node, deconstructable as (from_node, current)
        """
        return heapq.heappop(self.open)

    def _wrap(self, node_from, current):
        return self.Node(node_from, current, self.get_comparable(current))


class SemiRandomDijkstraSavingAllRoutes(SemiRandomDijkstra):
    # open is a heap of nodes sorted by their comparables
    # closed is a dictionary of nodes to, with values a list of nodes from
    class Node(SemiRandomDijkstra.Node):
        def __init__(self, current, new, current_comparable):
            super().__init__([current], new, current_comparable)

        def append(self, current):
            self.current.append(current)

    def add_edge_to_closed(self, current, new):
        assert new not in self.closed
        assert current is None or isinstance(current, list)

        self.closed[new] = current

    def update_closed(self, current, new):
        if new in self.closed:
            self.closed[new].append(current)
            return True
        return False

    def _wrap(self, node_from, current):
        return self.Node(node_from, current, self.get_comparable(current))

    def get_from(self, end):
        """
        :return: The first node from which can be walked to the specified node
        """
        return self.closed[end][0]

    def find_random_routes(self):
        """
        Returns random routes from start nodes to destination nodes, ad infinitum.
        """
        # if the algorithm hasn't run (to completion) yet, run it
        if len(self.closed) == 0 or len(self.open) != 0:
            for _ in self.find_route():
                pass

        multiplicities = self._get_all_closed_multiplicities()
        destinations = list(filter(self._is_dest, self.closed.keys()))

        if len(destinations) != 0:
            while True:
                route = []
                nodes_from = destinations
                while nodes_from[0]:
                    chosen_node = random.choices(nodes_from, list(map(multiplicities.__getitem__, nodes_from)))[0]
                    route.append(chosen_node)
                    nodes_from = self.closed[route[-1]]
                yield tuple(route)

    def _get_all_closed_multiplicities(self):
        """
        Replaces all closed nodes by named tuples with the attribute 'multiplicity', signifying the number of ways
        a route can be made to end at the accompanying node.
        """
        get_multiplicity = {}

        def compute_multiplicity(node):
            if node not in get_multiplicity:
                result = 1
                for from_node in self.closed[node]:
                    if from_node is None:
                        break
                    if from_node not in get_multiplicity:
                        compute_multiplicity(from_node)
                    result *= get_multiplicity[from_node]
                get_multiplicity[node] = result

        while len(get_multiplicity) != len(self.closed):
            for to_node, _ in self.closed.items():
                compute_multiplicity(to_node)

        return get_multiplicity


def all_slotwise_combinations(collections_: List[List]):
    indices = [0 for _ in collections_]

    while True:
        yield tuple(collection[i] for i, collection in zip(indices, collections_))
        for i in range(len(collections_) + 1):
            if indices[i] + 1 == len(collections_[i]):
                if i + 1 == len(collections_):
                    return
                indices[i] = 0
            else:
                indices[i] += 1
                break
