from chromokeras import *
import unittest
from keras.layers import *
from distribution import *


class TestChromokeras(unittest.TestCase):
    def test_simple(self):
        distribution = CollectionDistribution([10, 20, 50, 100])
        chromosome = Chromosome.create([ParameterAllele(Dense, units=(10, distribution))])

        result = chromosome_to_nn(chromosome)

        self.assertIsNotNone(result)

    def test_cache(self):
        distribution = CollectionDistribution([10, 20, 50, 100])
        chromosome = Chromosome.create([ParameterAllele(Dense, units=(10, distribution))])

        result1 = chromosome_to_nn(chromosome)
        result2 = chromosome_to_nn(chromosome)

        self.assertEqual(result1, result2)

    def test_cache_not_used_if_different(self):
        distribution = CollectionDistribution([10, 20, 50, 100])
        chromosome1 = Chromosome.create([ParameterAllele(Dense, units=(10, distribution))])
        chromosome2 = Chromosome.create([ParameterAllele(Dense, units=(20, distribution))])

        result1 = chromosome_to_nn(chromosome1)
        result2 = chromosome_to_nn(chromosome2)

        self.assertNotEqual(result1, result2)
