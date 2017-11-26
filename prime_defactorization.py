import operator
import random
import unittest
from functools import reduce
from math import gcd
from typing import *
from itertools import *
import random_dijkstra


def flatten(lists):
    return [item for sublist in lists for item in sublist]


def skip_first_and_last(seq):
    first = True
    second = True
    previous = None
    for elem in seq:
        if second:
            if first:
                first = False
            else:
                second = False
        else:
            yield previous
        previous = elem


def brute_force_prime_factors(n, limit=None):
    """
    Brute-forces the calculation of all factors of the specified number.
    :param n: The number to factorize.
    :param limit: Yields primes up to this number, and the remainder is then also yielded.
    """

    assert isinstance(n, int)
    assert isinstance(limit, int) or not limit

    i = 2
    factors = []
    while i * i <= n and (not limit or i <= limit):
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def prime_factors(n, algo: Callable[[int], int], limit=100):
    """
    :param n: The number to factorize.
    :param algo: An algorithm finding a factor of its argument.
    :param limit: Uses brute-force to find the factors up to this number.
    """
    if limit:
        result = brute_force_prime_factors(n, limit)
        if result[-1] > limit:
            n = result[-1]
            del result[-1]
        else:
            return result
    else:
        result = []

    factor = None
    while factor != n and n != 1:
        factor = algo(n)
        result.append(factor)
        n //= factor
    result.sort()
    return result


# noinspection PyPep8Naming
def pollardRho(n):
    """
    Finds a factor of n.
    """
    if n % 2 == 0:
        return 2
    x = random.randint(1, n - 1)
    y = x
    c = random.randint(1, n - 1)
    g = 1
    while g == 1:
        x = ((x * x) % n + c) % n
        y = ((y * y) % n + c) % n
        y = ((y * y) % n + c) % n
        g = gcd(abs(x - y), n)
    return g


def brent(n):
    """
    Finds a factor of n.
    """
    ys, x = None, None
    if n % 2 == 0:
        return 2
    y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), random.randint(1, n - 1)
    g, r, q = 1, 1, 1
    while g == 1:
        x = y
        for i in range(r):
            y = ((y * y) % n + c) % n
        k = 0
        while k < r and g == 1:
            ys = y
            for i in range(min(m, r - k)):
                y = ((y * y) % n + c) % n
                q = q * (abs(x - y)) % n
            g = gcd(q, n)
            k = k + m
        r = r * 2
    if g == n:
        while True:
            ys = ((ys * ys) % n + c) % n
            g = gcd(abs(x - ys), n)
            if g > 1:
                break
    return g


def _miller_rabin_pass(a, s, d, n):
    a_to_power = pow(a, d, n)
    if a_to_power == 1:
        return True
    for _ in range(s - 1):
        if a_to_power == n - 1:
            return True
        a_to_power = (a_to_power * a_to_power) % n
    return a_to_power == n - 1


def miller_rabin(n):
    """
    Determines whether n is prime.
    """
    d = n - 1
    s = 0
    while d % 2 == 0:
        d >>= 1
        s += 1

    for _ in range(20):
        a = 0
        while a == 0:
            a = random.randrange(n)
        if not _miller_rabin_pass(a, s, d, n):
            return False
    return True


def all_primes(n):
    """
    Yields all primes up to n.
    """
    if n <= 2:
        return []
    sieve = [True] * (n + 1)
    for x in range(3, int(n ** 0.5) + 1, 2):
        for y in range(3, (n // x) + 1, 2):
            sieve[x * y] = False

    return [2] + [i for i in range(3, n, 2) if sieve[i]]


def _to_multiplicities(primes):
    assert primes == sorted(primes)

    uniques = list(sorted(set(primes)))
    counts = list(map(lambda x: sum(1 for _ in filter(lambda p: p == x, primes)), uniques))
    return uniques, counts


def defactorize(n: Union[int, Iterable[int]], combination_length: int):
    """
    :param n: The number or numbers to divide into primes.
    :param combination_length: The length of the combinations to yield.
    :return: All unique combinations of the specified length of numbers that together multiply to the product of n.
    """

    if isinstance(n, int):
        n = [n]

    primes = flatten(prime_factors(n_, brent) for n_ in n)
    primes, multiplicities = _to_multiplicities(primes)
    explicit_multiplicities = list(map(lambda i: list(range(i + 1)), multiplicities))

    def subtract(collection, chosen):
        assert len(collection) == len(chosen)

        return list(list(iter(islice(pc, 0, len(pc) - chosen_count))) for pc, chosen_count in zip(collection, chosen))

    def combine(remaining, rank_remaining):
        if rank_remaining == 0:
            yield tuple(rem[-1] for rem in remaining),
        # if you don't want to exclude buckets containing 1, remove the call skip_first_and_last
        for selection in skip_first_and_last(random_dijkstra.all_slotwise_combinations(remaining)):
            for combi in combine(subtract(remaining, selection), rank_remaining - 1):
                yield (selection,) + combi

    def to_number(selection):
        return reduce(operator.mul, (primes[i] ** multiplicity for i, multiplicity in enumerate(selection)))

    for combination in combine(explicit_multiplicities, combination_length - 1):
        result = [to_number(powers) for powers in combination]
        yield result


def defactorize_random_access(n: Union[int, Iterable[int]], combination_length: int, i: int):
    """
    :param n: The number or numbers to divide into primes.
    :param combination_length: The length of the combinations to yield.
    :param i: The random access index.
    :return: The unique combination of the specified length of numbers that together multiply to the product of n
    at the specified index in the sequence generated by defactorize.
    """

    islice(defactorize(n, combination_length), i, i)





class Tests(unittest.TestCase):
    def test_pollard_rho_finds_unique_primes(self):
        x = set((tuple(prime_factors(9843119861, pollardRho)) for _ in range(1000)))
        self.assertEqual(len(x), 1)

    def test_defactorize_12_over_2_buckets(self):
        x = list(defactorize(12, 2))

        self.assertSequenceEqual(x, [[2, 6], [4, 3], [3, 4], [6, 2]])

    def test_defactorize_12_over_3_buckets(self):
        x = list(defactorize(12, 3))

        self.assertSequenceEqual(x, [[2, 2, 3], [2, 3, 2], [3, 2, 2]])

    def test_skip_first_and_last(self):
        result = list(skip_first_and_last([-1, 1, 5, 2, 3]))

        self.assertSequenceEqual(result, [1, 5, 2])


if __name__ == '__main__':
    Tests().test_defactorize_12_over_2_buckets()
