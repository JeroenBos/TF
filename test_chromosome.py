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
        self.assertEqual(len(routes), len(DenseBuilder.default_distributions['units'].collection))

    def test_find_random_routes_impossible(self):
        builder = ChromokerasBuilder((10, 10), (10,), [DenseBuilder()])

        routes = list(islice(builder.find_random_routes(2), 100))
        self.assertEqual(len(routes), 0)


if __name__ == '__main__':
    unittest.main()
