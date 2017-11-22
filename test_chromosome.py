import unittest
import hyperchrom as c
from chromokeras import *
import keras
from keras.activations import *
from distribution import *


# noinspection PyMethodMayBeStatic
class ChromosomeTests(unittest.TestCase):
    def test_collection(self):
        builder = c.ParameterAlleleBuilder(keras.layers.Dense, units=CollectionDistribution([10, 20, 50, 100]))
        allele = builder.create(units=10)

        crossed_result = builder.crossover(allele, allele)

        assert allele == crossed_result

    def test_crossover_result_is_between(self):
        builder = c.ParameterAlleleBuilder(keras.layers.Dense, units=CollectionDistribution([10, 20, 50, 100]))
        allele1 = builder.create(units=10)
        allele2 = builder.create(units=50)
        possible_outcomes = [allele1, allele2, builder.create(units=20)]

        for _ in range(20):
            crossed_result = builder.crossover(allele1, allele2)

            assert crossed_result in possible_outcomes

    def test_crossover_activation_functions_hashability(self):
        builder = c.ParameterAlleleBuilder(keras.layers.Dense, units=CollectionDistribution([relu, sigmoid, tanh]))
        allele1 = builder.create(units=relu)
        allele2 = builder.create(units=tanh)

        crossed_result = builder.crossover(allele1, allele2)

        assert crossed_result

    def test_chromosome_reference_equality(self):
        allele1 = c.ParameterAllele.create(None, keras.layers.Dense, units=10)

        chromosome1 = c.Chromosome.create([allele1])
        chromosome2 = c.Chromosome.create([allele1])

        assert chromosome1 is chromosome2

    def test_chromosome_negated_reference_equality(self):
        distribution = CollectionDistribution([10, 20, 50, 100])
        allele_builder = c.ParameterAlleleBuilder(keras.layers.Dense, units=distribution)
        allele1 = allele_builder .create(units=10)
        allele2 = allele_builder.create(units=20)

        chromosome1 = c.Chromosome.create([allele1])
        chromosome2 = c.Chromosome.create([allele2])

        assert chromosome1 is not chromosome2

    def test_invalid_parameter_throws(self):
        with self.assertRaises(AttributeError):
            c.ParameterAllele(lambda: None, does_not_exist=0)

    def test_allele_mutation_possibility_count(self):
        distribution = CollectionDistribution([10, 20, 50, 100])
        allele = c.ParameterAlleleBuilder(keras.layers.Dense, units=distribution, something_else=distribution)

        self.assertEqual(4 * 4 - 1, allele.cumulative_mutation_weight)

    def test_chromosome_mutation_possibility_count(self):
        distribution = CollectionDistribution([10, 20, 50, 100])
        allele_builder = c.ParameterAlleleBuilder(keras.layers.Dense, units=distribution, something_else=distribution)
        allele = allele_builder.create(units=50)
        chromosome_builder = c.ChromosomeBuilder([allele_builder])

        chromosome = chromosome_builder.create([allele, allele])

        self.assertEqual(4 * 4 - 1 + 4 * 4 - 1, chromosome_builder.get_small_mutation_weight(chromosome))

    def test_genome_initialization(self):
        allele = c.ParameterAllele.create(None, keras.layers.Dense, units=10)
        chromosome = c.Chromosome.create([allele])

        genome = c.Genome.create([chromosome])

        self.assertTrue(len(genome.chromosomes), 1)

    def test_genome_mutation(self):
        distribution = CollectionDistribution([10, 20, 50, 100])
        allele_builder = c.ParameterAlleleBuilder(keras.layers.Dense, units=distribution)
        allele = allele_builder.create(units=10)
        chromosome = c.Chromosome.create([allele])
        genome = c.Genome.create([chromosome])

        genome_builder = c.GenomeBuilder(c.ChromosomeBuilder([allele_builder]), large_mutation_probability=0)
        mutated_genome = genome_builder.mutate(genome)

        self.assertIsNot(genome, mutated_genome)

    def test_find_random_routes_one_diversion(self):
        builder = ChromokerasBuilder((10,), (10,), [DenseBuilder()])

        routes = set(islice(builder.find_random_routes(end=2), 100))
        self.assertEqual(len(routes), len(DenseBuilder.default_distributions['units'].get_collection()))

    def test_find_random_routes_impossible(self):
        builder = ChromokerasBuilder((10, 10), (10,), [DenseBuilder()])

        routes = list(islice(builder.find_random_routes(2), 100))
        self.assertEqual(len(routes), 0)

    def test_find_random_routes_impossible_conv2D(self):
        random.seed(999)
        builder = ChromokerasBuilder((10, 10, 1), (10,), [DenseBuilder(), Conv2DBuilder()])

        routes = list(islice(builder.find_random_routes(2), 100))
        self.assertEqual(len(routes), 0)  # doesn't work because flatten isn't included

    def test_find_only_flatten_dense_possibility(self):
        builder = ChromokerasBuilder((10, 10, 1), (100,), [DenseBuilder(), Conv2DBuilder(), FlattenBuilder()])

        routes = list(set(islice(builder.find_random_routes(1), 100)))
        self.assertEqual(len(routes), 1)  # there is only one route, namely flatten immediately and then a Dense(100)

    def test_find_only_flatten_dense_dense_possibility(self):
        builder = ChromokerasBuilder((10, 10, 1), (100,), [DenseBuilder(), FlattenBuilder()])

        routes = list(set(islice(builder.find_random_routes(2), 100)))
        # there is only one type of route, namely flatten immediately and then a Dense(X) and then Dense(100)
        self.assertEqual(len(routes), len(DenseBuilder.default_distributions['units']))

    def test_find_only_conv2d_dense_possibility(self):
        builder = ChromokerasBuilder((10, 10, 1), (36 * 8,), [Conv2DBuilder(kernel_size=SetDistribution([(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)])), FlattenBuilder()])

        routes = list(set(islice(builder.find_random_routes(1), 100)))
        # there is only one type of route, namely Conv2D(kernel_size = (5, 5)) and then flatten
        self.assertEqual(len(routes), 1)

    def test_conv2D_output_shape(self):
        filters = 11               # this test will work for
        kernel_size = (3, 1)       # arbitrary positive numbers here
        input_shape = (20, 13, 9)  # where kernel_size[i] < input_shape[i]     i = 1,2
        model = keras.models.Sequential(layers=[Conv2D(filters=filters, kernel_size=kernel_size, input_shape=input_shape)])

        expected = model.output_shape[1:]

        result = Conv2DBuilder.output_shape(filters=filters, kernel_size=kernel_size, input_shape=input_shape)

        self.assertEqual(expected, result)

    def test_conv2D_contains_only_input_rank_3(self):
        self.assertFalse(Conv2DBuilder.contains_input_rank(0))
        self.assertFalse(Conv2DBuilder.contains_input_rank(1))
        self.assertFalse(Conv2DBuilder.contains_input_rank(2))
        self.assertTrue(Conv2DBuilder.contains_input_rank(3))
        self.assertFalse(Conv2DBuilder.contains_input_rank(4))


if __name__ == '__main__':
    ChromosomeTests().test_find_only_conv2d_dense_possibility()
    unittest.main()
