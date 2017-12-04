from typing import *


class IntegerInterval:
    empty = None
    def __init__(self, interval: Union[int, List[int], Tuple[int, int]]):
        """
        :param interval: A number, list, or range (by tuple, upperbound inclusive)
        """

        assert isinstance(interval, (int, list, tuple))
        assert isinstance(interval, int) or all(isinstance(val, int) for val in iter(interval))
        assert isinstance(interval, (int, list)) or len(interval) == 2
        assert isinstance(interval, (int, tuple)) or list(sorted(interval)) == interval

        if isinstance(interval, tuple) and interval[0] > interval[1]:
            interval = interval[1], interval[0]

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

    def __len__(self):
        if isinstance(self.interval, int):
            return 1
        elif isinstance(self.interval, list):
            return len(self.interval)
        else:
            return self.interval[1] - self.interval[0] + 1  # 1 for upper bound being inclusive

    def __iter__(self):
        if isinstance(self.interval, int):
            return iter([self.interval])
        if isinstance(self.interval, list):
            return iter(self.interval)

        return iter(range(self.interval[0], self.interval[1] + 1))

    def __repr__(self):
        if isinstance(self.interval, int):
            return f'[{self.interval}]'
        if isinstance(self.interval, list):
            return repr(self.interval)

        return f'[{self.interval[0]}...{self.interval[1]}]'

    def intersection(self, other):
        assert isinstance(other, IntegerInterval)

        if isinstance(self.interval, int):
            if self.interval in other:
                return self
            else:
                return __class__.empty
        elif isinstance(self.interval, list):
            if isinstance(other.interval, int):
                return other.intersection(self)
            else:  # noinspection PyTypeChecker
                return IntegerInterval([i for i in self.interval if i in other])
        else:
            if not isinstance(other.interval, tuple):
                return other.intersection(self)
            else:
                min_ = max(self.interval[0], other.interval[0])
                max_ = min(self.interval[1], other.interval[1])  # inclusive
                if max_ < min_:
                    return __class__.empty
                else:
                    return IntegerInterval((min_, max_))

    def __eq__(self, other):
        assert isinstance(other, IntegerInterval)

        if self.interval == other.interval:
            return True
        if len(self) != len(other):
            return False

        if isinstance(self.interval, int):
            if isinstance(other.interval, list):
                return self.interval == other.interval[0]
            elif isinstance(other.interval, tuple):
                return self.interval == other.interval[0]
        elif isinstance(self.interval, list):
            if isinstance(other.interval, tuple):
                return all(elem in other.interval for elem in self)
        else:
            if isinstance(other.interval, list):
                return other == self
        return False

    def __hash__(self):
        # noinspection PyTypeChecker
        return hash(self.interval) if not isinstance(self.interval, list) else sum(i for i in self.interval)


IntegerInterval.empty = IntegerInterval([])
