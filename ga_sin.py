from ga import *
import random
from math import *
import visualization
from keras.layers import Dense
from keras.models import Sequential, clone_model
from keras.activations import tanh
from keras.losses import mean_squared_error
from keras.optimizers import SGD
from keras.initializers import glorot_uniform
import keras

_X_SCALE = .2
_LENGTH = 10


def _create_archetype():
    model = Sequential()
    model.add(Dense(20, activation=tanh, input_shape=(1,), ))
    model.add(Dense(20, activation=tanh))
    model.add(Dense(1, activation=tanh))
    model.compile(optimizer=SGD(0.1), loss=mean_squared_error)
    return model


def count_relevant_parameters(model):
    result = 0
    for layer in model.layers:
        assert isinstance(layer, Dense)
        result += layer.get_weights()[0].size
    return result


class Member:
    archetype = _create_archetype()
    archetype_shape = (len(archetype.get_weights()),) if isinstance(archetype.get_weights(),
                                                                    list) else archetype.get_weights().shape
    archetype_relevant_params_count = count_relevant_parameters(archetype)

    def __init__(self, model):
        self.__model = model

    def mutate(self):
        weight_index = random.randint(0, Member.archetype_relevant_params_count - 1)

        debug = False
        for layer in self.__model.layers:
            if isinstance(layer, Dense):
                relevant_layer_length = layer.get_weights()[0].size
                if weight_index >= relevant_layer_length:
                    weight_index -= relevant_layer_length
                    continue
                else:
                    weights = layer.get_weights()
                    i = np.unravel_index((weight_index,), weights[0].shape)
                    weights[0][i] = Member.mutate_weight(weights[0][i])
                    layer.set_weights(weights)
                    debug = True
        assert debug

    @staticmethod
    def mutate_weight(w):
        return random.gauss(0, 1) * abs(w)

    def clone(self):
        return Member(clone_model(self.__model))

    def fitness(self):
        predictions = self.predict()
        return sum((val - sin(i * _X_SCALE)) ** 2 for i, val in enumerate(predictions))

    def predict(self):
        return self.__model.predict([i * _X_SCALE for i in range(_LENGTH)])

    @staticmethod
    def crossover(a, b):
        child = Member.create_random()
        for a_layer, b_layer, cloned_layer in zip(a.__model.layers, b.__model.layers, child.__model.layers):
            if isinstance(a_layer, Dense):
                cloned_layer.set_weights(array_crossover(a_layer.get_weights(), b_layer.get_weights()))
        return child

    @staticmethod
    def create_random():
        result = Member(clone_model(Member.archetype))
        for layer in result.__model.layers:
            if isinstance(layer, Dense):
                new_weights = glorot_uniform()(layer.weights[0].shape.as_list()).eval(
                    session=keras.backend.get_session())
                new_weights = [new_weights] + layer.get_weights()[1:]
                layer.set_weights(new_weights)
        return result

    def __repr__(self):
        return 'member fitness = ' + str(self.fitness())


gen_best_fitnesses = []


def callback(_generation, population, _fitnesses):
    gen_best_fitnesses.append(population[0].fitness())
    visualization.plot(plot, pts=(list(range(len(gen_best_fitnesses))), gen_best_fitnesses))
    visualization.plot(plot_predictions, subplot_index=(1, 2), predictions=population[0].predict())


def plot(subplot, pts):
    subplot.scatter(pts[0], pts[1])


def plot_predictions(subplot, predictions):
    subplot.scatter([range(_LENGTH)], predictions)
    subplot.scatter([range(_LENGTH)], [sin(i * _X_SCALE) for i in range(_LENGTH)])


def print_gen_number(generation_index, *_args):
    print(generation_index)


if __name__ == '__main__':
    ga(10,
       Member.fitness,
       Member.create_random,
       Member.mutate,
       Member.crossover,
       Member.clone,
       crossover_fraction=0.5,
       callbacks=[callback, print_gen_number])
