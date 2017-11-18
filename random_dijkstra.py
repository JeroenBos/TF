from typing import *
import heapq
import random
import sys
import collections

T = TypeVar('T')


class RandomWalk:
    """
        Gets a route by walking randomly from the root of a tree to any destination node.
    """

    def __init__(self,
                 start: Iterable[T],
                 get_neighbors: Callable[[T], Iterable],
                 f_is_dest: Callable[[T], bool]):
        assert all(isinstance(start_, collections.Hashable) for start_ in start)
        self.open = self.create_open(start)
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
                was_already_in_closed = self.update_closed(current, new)
                if not was_already_in_closed:
                    self.push(current, new)

    def _generate_result(self, end):
        while end is not None:
            yield end
            end = self.get_from(end)

    def get_from(self, end):
        """
        :return: The node before the specified node
        """
        return self.closed[end]



class RandomWalkSavingAllRoutes(RandomWalk):
    """
        Gets a route by walking randomly from the root of a tree to any destination node.
    """
    def push(self, current, new):
        self.open[new] = current

    def add_edge_to_closed(self, current, new):
        self.closed[new] = [current]

    def update_closed(self, current, new):
        if new in self.closed:
            self.closed[new].append(current)
            return False
        return True

    def get_from(self, end):
        """
        :return: The node before the specified node
        """
        return self.closed[end][0]


class Dijkstra(RandomWalk):
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

    def __init__(self, 
                 start,
                 get_neighbors,
                 f_is_dest,
                 get_comparable: Callable[[T], object]):
        self.get_comparable = get_comparable
        super().__init__(start, get_neighbors, f_is_dest)

    def create_open(self, start):
        heap = [self.Node(None, item, self.get_comparable(item)) for item in start]
        heapq.heapify(heap)
        return heap

    def push(self, current, new):
        heapq.heappush(self.open, self.Node(current, new, self.get_comparable(new)))

    def pop(self):
        return heapq.heappop(self.open)

