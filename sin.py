from math import sin, pi
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from hypermin import *
from hyperopt import hp
from hyperopt.pyll import scope
import persistence
from Visualization import OneDValidationPlotCallback

directory = "D:\\TFlogs\\"
INPUT_SIZE = 1000
DOMAIN_MAX = 2 * pi

sin_input = np.array([[i * DOMAIN_MAX / INPUT_SIZE] for i in range(INPUT_SIZE)])
sin_output = np.array([sin(x) for x in sin_input])

space = {'choice': hp.choice('num_layers',
                             [{'num_layers': 2, 'units2': 1},
                              {'num_layers': 3, 'units2': 50 * scope.int(hp.quniform('units2', 1, 10, 1)),
                                                'units3': 1}
                              ]),

         'units1': 50 * scope.int(hp.quniform('units1', 1, 10, 1)),

         'epochs': 10,
         'optimizer': hp.choice('optimizer', [SGD()]),
         'activation': keras.activations.sigmoid,
         'loss': keras.losses.mean_squared_error
         }


def create_model(params, input_dim):
    persistence.print_param_names(params)

    model = persistence.try_find(params, directory)
    if model:
        print('model loaded')
        model.parameters = params
        return model

    model = Sequential()
    model.parameters = params
    model.add(Dense(units=params['units1'], input_dim=input_dim))
    model.add(Activation(params['activation']))

    model.add(Dense(units=params['units2']))
    model.add(Activation(params['activation'] if params['num_layers'] == 3 else keras.activations.tanh))

    if params['num_layers'] == 3:
        model.add(Dense(units=params['units3']))
        model.add(Activation(keras.activations.tanh))

    model.compile(optimizer=params['optimizer'], loss=params['loss'])
    return model


if __name__ == '__main__':
    hypermin(space, create_model, sin_input, sin_output, sin_input, sin_output,
             verbose=0,
             callbacks=[keras.callbacks.TensorBoard(directory),
                        persistence.Save(directory),
                        OneDValidationPlotCallback(sin_input, sin_output)])
