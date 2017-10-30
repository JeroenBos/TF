from math import sin, pi
from itertools import islice
import tensorflow as tf
import numpy as np
import keras.initializers
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
from hypermin import *
import persistence
from Visualization import OneDValidationContinuousPlotCallback
from persistence import *

LOG_DIRECTORY = "D:\\TFlogs\\"
DX = 0.01
INPUT_SIZE = 100


def generate_input_and_output():
    x_ = np.random.uniform(0, 2 * pi)
    for i in range(INPUT_SIZE):
        yield [[sin(x_)], [sin(x_ + DX)]], sin(x_ + DX + DX)
        x_ += DX


x, y = (np.array(t) for t in zip(*islice(generate_input_and_output(), 0, INPUT_SIZE)))
assert x.shape == (INPUT_SIZE, 2, 1)
space = {'units1': 20,
         'units2': 20,
         'learning_rate': 0.1,
         'loss': keras.losses.mean_squared_error,
         'epochs': 10000
         }


def create_model(params, input_shape):
    persistence.print_param_names(params)

    model = persistence.try_find(params, LOG_DIRECTORY, verbose=1)
    if model is None:
        model = Sequential()
        model.add(SimpleRNN(units=params['units1'],
                            kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
                            recurrent_initializer=keras.initializers.Identity(gain=1.0),
                            input_shape=input_shape))
        print(model.output_shape)
        model.add(Dense(units=params['units2'], activation=keras.activations.tanh))
        model.add(Dense(units=1, activation=keras.activations.tanh))

        model.compile(optimizer=SGD(params['learning_rate']), loss=params['loss'])
    return model


if __name__ == '__main__':
    callbacks = [TensorBoardSummaryScalars({'learning_rate': lambda model: model.optimizer.lr}),
                 CustomTensorBoardSummary({'units1': lambda model: model.layers[0].units,
                                           'units2': lambda model: model.layers[2].units}),
                 keras.callbacks.TensorBoard(LOG_DIRECTORY),
                 ]

    hypermin(space, create_model, x, y, x, y, verbose=0, callbacks=callbacks)
