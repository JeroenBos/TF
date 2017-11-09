import unittest
import hyperchrom as c
import keras


class ChromosomeTests(unittest.TestCase):
    def test_collection(self):
        distribution = c.ParameterAllele.CollectionDistribution([10, 20, 50, 100])
        allele = c.ParameterAllele(keras.layers.Dense, units=(10, distribution))

        crossed_result = allele.crossover(allele)

        assert allele == crossed_result


if __name__ == '__main__':
    unittest.main()
