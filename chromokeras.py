from hyperchrom import *
from keras.layers import *
from itertools import *

_cached_nns = {}


def _create_layer(allele: ParameterAllele):
    return allele.layer_type(**allele.parameters)


def genome_to_nn(genome: Genome):
    return map(chromosome_to_nn, genome.chromosomes)


def chromosome_to_nn(chromosome: Chromosome):
    layers = []
    for allele in chromosome.alleles:
        assert isinstance(allele, ParameterAllele)
        try:
            nn_list = _cached_nns[allele]
            layer = next(nn for nn in nn_list if nn not in layers)
        except KeyError:
            layer = _create_layer(allele)
            _cached_nns[allele] = [layer]
        except StopIteration:
            layer = _create_layer(allele)
            _cached_nns[allele].append(layer)
        layers.append(layer)
    return layers


class ChromokerasBuilder(ChromosomeBuilder):
    def __init__(self):
        super().__init__()


    def generate(self):
        pass

    def mutate_large(self, chromosome: Chromosome):
        pass
