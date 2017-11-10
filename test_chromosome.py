import unittest
import hyperchrom as c
import keras
from keras.activations import *


# noinspection PyMethodMayBeStatic
class ChromosomeTests(unittest.TestCase):
    def test_collection(self):
        distribution = c.ParameterAllele.CollectionDistribution([10, 20, 50, 100])
        allele = c.ParameterAllele(keras.layers.Dense, units=(10, distribution))

        crossed_result = allele.crossover(allele)

        assert allele == crossed_result

    def test_crossover_result_is_between(self):
        distribution = c.ParameterAllele.CollectionDistribution([10, 20, 50, 100])
        allele1 = c.ParameterAllele(keras.layers.Dense, units=(10, distribution))
        allele2 = c.ParameterAllele(keras.layers.Dense, units=(50, distribution))
        possible_outcomes = [allele1, allele2, c.ParameterAllele(keras.layers.Dense, units=(20, distribution))]

        for _ in range(20):
            crossed_result = allele1.crossover(allele2)

            assert crossed_result in possible_outcomes

    def test_crossover_activation_functions_hashability(self):
        distribution = c.ParameterAllele.SetDistribution([relu, sigmoid, tanh, linear])

        allele1 = c.ParameterAllele(keras.layers.Dense, units=(relu, distribution))
        allele2 = c.ParameterAllele(keras.layers.Dense, units=(tanh, distribution))

        crossed_result = allele1.crossover(allele2)

        assert crossed_result


if __name__ == '__main__':
    unittest.main()
