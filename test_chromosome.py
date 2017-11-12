import unittest
import hyperchrom as c
import keras
from keras.activations import *
from distribution import *


# noinspection PyMethodMayBeStatic
class ChromosomeTests(unittest.TestCase):
    def test_collection(self):
        distribution = CollectionDistribution([10, 20, 50, 100])
        allele = c.ParameterAllele(keras.layers.Dense, units=(10, distribution))

        crossed_result = allele.crossover(allele)

        assert allele == crossed_result

    def test_crossover_result_is_between(self):
        distribution = CollectionDistribution([10, 20, 50, 100])
        allele1 = c.ParameterAllele(keras.layers.Dense, units=(10, distribution))
        allele2 = c.ParameterAllele(keras.layers.Dense, units=(50, distribution))
        possible_outcomes = [allele1, allele2, c.ParameterAllele(keras.layers.Dense, units=(20, distribution))]

        for _ in range(20):
            crossed_result = allele1.crossover(allele2)

            assert crossed_result in possible_outcomes

    def test_crossover_activation_functions_hashability(self):
        distribution = SetDistribution([relu, sigmoid, tanh, linear])

        allele1 = c.ParameterAllele(keras.layers.Dense, units=(relu, distribution))
        allele2 = c.ParameterAllele(keras.layers.Dense, units=(tanh, distribution))

        crossed_result = allele1.crossover(allele2)

        assert crossed_result

    def test_chromosome_reference_equality(self):
        distribution = CollectionDistribution([10, 20, 50, 100])
        allele1 = c.ParameterAllele(keras.layers.Dense, units=(10, distribution))

        chromosome1 = c.Chromosome.create([allele1])
        chromosome2 = c.Chromosome.create([allele1])

        assert chromosome1 is chromosome2

    def test_chromosome_negated_reference_equality(self):
        distribution = CollectionDistribution([10, 20, 50, 100])
        allele1 = c.ParameterAllele(keras.layers.Dense, units=(10, distribution))
        allele2 = c.ParameterAllele(keras.layers.Dense, units=(100, distribution))

        chromosome1 = c.Chromosome.create([allele1])
        chromosome2 = c.Chromosome.create([allele2])

        assert chromosome1 is not chromosome2

    def test_invalid_parameter_throws(self):
        with self.assertRaises(AttributeError):
            c.ParameterAllele(lambda: None, does_not_exist=(0, CollectionDistribution([0])))

    def test_allele_mutation_possibility_count(self):
        distribution = CollectionDistribution([10, 20, 50, 100])
        allele = c.ParameterAllele(keras.layers.Dense, units=(10, distribution), something_else=(20, distribution))

        self.assertEqual(4 * 4 - 1, allele.cumulative_mutation_count)

    def test_chromosome_mutation_possibility_count(self):
        distribution = CollectionDistribution([10, 20, 50, 100])
        allele = c.ParameterAllele(keras.layers.Dense, units=(10, distribution), something_else=(20, distribution))

        chromosome = c.Chromosome.create([allele, allele])

        self.assertEqual(4 * 4 - 1 + 4 * 4 - 1, chromosome.cumulative_mutation_count)

    def test_genome_initialization(self):
        distribution = CollectionDistribution([10, 20, 50, 100])
        allele = c.ParameterAllele(keras.layers.Dense, units=(10, distribution))
        chromosome = c.Chromosome.create([allele])

        genome = c.Genome([chromosome])

        self.assertTrue(len(genome.chromosomes), 1)

    def test_genome_mutation(self):
        distribution = CollectionDistribution([10, 20, 50, 100])
        allele = c.ParameterAllele(keras.layers.Dense, units=(10, distribution))
        chromosome = c.Chromosome.create([allele])
        genome = c.Genome([chromosome])

        mutated_genome = genome.mutate_small()

        self.assertIsNot(genome, mutated_genome)



if __name__ == '__main__':
    ChromosomeTests().test_genome_mutation()
    unittest.main()
