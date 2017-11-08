from typing import List, Tuple
import keras
import itertools
import random


class GenePool:
    def __init__(self, space, fitness, max_learnable_params=-1):
        assert isinstance(space, dict)
        for key, value in space.items:
            assert key.__module__ in [keras.layers, keras.activations]
            assert value is None or isinstance(value, tuple)
            for parameter in value:
                assert isinstance(parameter[0], str)
                for parameter_range in parameter[1:]:
                    assert isinstance(parameter_range, list) or isinstance(parameter_range, tuple)
        assert max_learnable_params == -1 or max_learnable_params > 0

        self.max_learnable_params = max_learnable_params
        self.space = space
        self.fitness = fitness

    def mutate(self, chromosome):
        pass

    def generate(self):
        pass

# immutable
class HyperAllele():
    def __init__(self, tag: str):
        pass

    def __hash__(self):
        raise NotImplementedError('must implement __hash__')

    def __eq__(self, other):
        raise NotImplementedError('must implement __eq__')

    def can_crossover_with(self):
        return False

    def crossover(self, other):
        raise NotImplementedError()



class HyperChromosome:
    all_hc = {}

    def __init__(self, alleles: List[HyperAllele], gene: GenePool):
        super().__init__()
        self.__alleles = alleles
        self.gene = gene

    @classmethod
    def create(cls, alleles: List[HyperAllele]):
        result = HyperChromosome(alleles)
        if result in cls.all_hc:
            result = cls.all_hc[result]
        else:
            cls.all_hc[result] = result
        return result

    def clone(self):
        return HyperChromosome([allele for allele in self.__alleles])

    @staticmethod
    def crossover(a, b):
        assert isinstance(a, HyperChromosome)
        assert isinstance(b, HyperChromosome)
        assert a != b

        def alleles_equal(alleles: Tuple[HyperAllele, HyperAllele]):
            return alleles[0] == alleles[1]

        head = list(allele for allele, _ in itertools.takewhile(alleles_equal, zip(a, b)))
        tail = list(allele for allele, _ in itertools.takewhile(alleles_equal, zip(a[-1::], b[-1::])))

        remaining = ((a[i], b[i]) for i in range(len(head), min(len(a), len(b)) - len(tail)))
        for allele_a, allele_b in remaining:
            if allele_a.can_crossover_with(allele_b):
                head.append(allele_a.crossover_with(allele_b))

        for allele_a, allele_b in reversed(list(remaining)):
            if allele_a.can_crossover_with(allele_b):
                tail.append(allele_a.crossover_with(allele_b))

        tail_start = -len(tail) if len(tail) != 0 else None
        remaining_a = a[len(head):tail_start]
        remaining_b = b[len(head):tail_start]

        middle = list(__class__._randomly_mix(remaining_a, remaining_b))

        return head + middle + tail

    @staticmethod
    def _randomly_mix(a: list, b: list):
        ai, bi = 0, 0

        while ai < len(a) or bi < len(b):
            if random.randint(0, len(a) + len(b)) > len(a):
                ai, bi = bi, ai
                a, b = b, a

            if ai < len(a):
                yield a[ai]
                ai += 1
            if random.uniform(0, 1) < len(b) / len(a):
                bi += 1




    def __hash__(self):
        return sum(hash(allele) for allele in self.__alleles)

    def __eq__(self, other):
        return isinstance(other, HyperChromosome) and self.__alleles == other.__alleles

    def __iter__(self):
        return self.__alleles

def ga(population_size, fitness, gene: GenePool, *callbacks):
    from ga import ga
    ga(population_size,
       fitness,
       gene.generate,
       gene.mutate,
       HyperChromosome.crossover,
       HyperChromosome.clone,
       callbacks)


if __name__ == '__main__':
    for _ in range(200):
        x = list(HyperChromosome._randomly_mix([0, 1, 2], [3, 4, 5, 6, 7, 8, 9]))
        print(x)



