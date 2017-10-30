from math import sin, pi
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
from hypermin import *
from hyperopt import hp
from hyperopt.pyll import scope
import persistence
from Visualization import OneDValidationContinuousPlotCallback
from persistence import *


LOG_DIRECTORY = "D:\\TFlogs\\"
INPUT_SIZE = 100
DOMAIN_MAX = 2*pi

sin_input = np.array([[i * DOMAIN_MAX / INPUT_SIZE] for i in range(INPUT_SIZE)])
sin_output = np.array([sin(x) for x in sin_input])

space = {'choice': hp.choice('num_layers',
                             [{'num_layers': 3, 'units2': 5 * scope.int(hp.quniform('units2', 1, 10, 1)),
                                                'units3': 1}
                              ]),

         'units1': 5 * scope.int(hp.quniform('units1', 1, 10, 1)),

         'epochs': 10000,
         'learning_rate': hp.choice('learning_rate', [0.1, 0.2]),
         'activation': keras.activations.tanh,
         'loss': keras.losses.mean_squared_error
         }


def create_model(params, input_shape):
    persistence.print_param_names(params)

    model = persistence.try_find(params, LOG_DIRECTORY, verbose=1)
    if model is None:
        model = Sequential()
        model.add(Dense(units=params['units1'], input_shape=input_shape))
        model.add(Activation(params['activation']))

        model.add(Dense(units=params['units2']))
        model.add(Activation(params['activation'] if params['num_layers'] == 3 else keras.activations.tanh))

        if params['num_layers'] == 3:
            model.add(Dense(units=params['units3']))
            model.add(Activation(keras.activations.tanh))

        model.compile(optimizer=SGD(params['learning_rate']), loss=params['loss'])
    return model


if __name__ == '__main__':
    callbacks = [TensorBoardSummaryScalars({'learning_rate': lambda model: model.optimizer.lr}),
                 CustomTensorBoardSummary({'units1': lambda model: model.layers[0].units,
                                           'units2': lambda model: model.layers[2].units}),
                 keras.callbacks.TensorBoard(LOG_DIRECTORY),
                 persistence.Save(LOG_DIRECTORY),
                 OneDValidationContinuousPlotCallback(sin_input, sin_output),
                 ReduceLROnPlateau('loss', patience=250, factor=0.8, min_lr=0.01)
                 ]
    hypermin(space, create_model, sin_input, sin_output, sin_input, sin_output, verbose=0, callbacks=callbacks)




