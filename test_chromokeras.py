from chromokeras import *
import unittest
from keras.layers import *
from distribution import *


class TestChromokeras(unittest.TestCase):
    def test_simple(self):
        chromosome = Chromosome.create([ParameterAllele.create(None, Dense, units=10)])

        result = chromosome_to_nn(chromosome)

        self.assertIsNotNone(result)

    def test_cache(self):
        chromosome = Chromosome.create([ParameterAllele.create(None, Dense, units=10)])

        result1 = chromosome_to_nn(chromosome)
        result2 = chromosome_to_nn(chromosome)

        self.assertEqual(result1, result2)

    def test_cache_not_used_if_different(self):
        chromosome1 = Chromosome.create([ParameterAllele.create(None, Dense, units=10)])
        chromosome2 = Chromosome.create([ParameterAllele.create(None, Dense, units=20)])

        result1 = chromosome_to_nn(chromosome1)
        result2 = chromosome_to_nn(chromosome2)

        self.assertNotEqual(result1, result2)

    def test_dense_default_creation(self):
        allele = DenseBuilder().create()

        self.assertEqual(allele.parameters['units'], 50)

    def test_conv2d_default_creation(self):
        allele = Conv2DBuilder().create()

        self.assertIsNotNone(allele)

    def test_ChromokerasBuilder_instantiation(self):
        cb = ChromokerasBuilder(input_shape=(10,), allele_builders=[DenseBuilder()])

        self.assertIsNotNone(cb)


    def test_set_simple_input_shape(self):
        cb = ChromokerasBuilder(input_shape=(10,), allele_builders=[DenseBuilder()])

