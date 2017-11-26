from typing import *


class IntegerInterval:
    empty = None
    def __init__(self, interval: Union[int, List[int], Tuple[int, int]]):
        """
        :param interval: A number, list, or range (by tuple, upperbound inclusive)
        """
        assert isinstance(interval, (int, list, tuple))
        assert isinstance(interval, int) or all(isinstance(val, int) for val in iter(interval))
        assert isinstance(interval, (int, list)) or (len(interval) == 2 and interval[0] <= interval[1])

        self.__interval = interval

    def __contains__(self, item):

        if isinstance(self.interval, int):
            return self.interval == item
        if isinstance(self.interval, list):
            return item in self.interval

        return self.interval[0] <= item <= self.interval[1]

    @property
    def interval(self):
        return self.__interval

    def __iter__(self):
        if isinstance(self.interval, int):
            return iter([self.interval])
        if isinstance(self.interval, list):
            return iter(self.interval)

        return range(self.interval[0], self.interval[1] + 1)

    def __repr__(self):
        if isinstance(self.interval, int):
            return f'[{self.interval}]'
        if isinstance(self.interval, list):
            return repr(self.interval)

        return f'[{self.interval[0]}...{self.interval[1]}]'


IntegerInterval.empty = IntegerInterval([])
