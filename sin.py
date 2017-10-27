from math import sin, pi
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import sys

INPUT_SIZE = 10
DOMAIN_MAX = 2 * pi

sin_input = np.array([[i * DOMAIN_MAX / INPUT_SIZE] for i in range(INPUT_SIZE)])
sin_output = np.array([sin(x) for x in sin_input])

space = {'choice': hp.choice('num_layers',
                             [{'layers': 'two',
                               'units2': 1},
                              {'layers': 'three',
                               'units2': hp.quniform('units2', 1, 5, 1),
                               'units3': 1}
                              ]),

         'units1': hp.quniform('units1', 1, 5, 1),

         'epochs': 100,
         'optimizer': hp.choice('optimizer', [SGD()]),
         'activation': keras.activations.sigmoid,
         'loss': keras.losses.mean_squared_error
         }


def create_model(params, input_dim):
    to_int(params, 'units1')
    to_int(params, 'units2', 'choice')
    print('Params testing: ', params)
    model = Sequential()
    model.add(Dense(units=params['units1'], input_dim=input_dim))
    model.add(Activation(params['activation']))

    model.add(Dense(units=1))
    model.add(Activation(params['activation']))

    if params['choice']['layers'] == 'three':
        model.add(Dense(units=1))
        model.add(Activation(params['activation']))

    model.compile(optimizer=params['optimizer'], loss=params['loss'])
    return model


def hypermin(space_, to_model, x, y, x_val, y_val, **kwargs):
    print("x.shape=" + str(x.shape))

    def f_nn(params):
        model = to_model(params, input_dim=x.shape[1])
        model.fit(x, y, epochs=params['epochs'], **kwargs)

        loss = model.evaluate(x=x_val, y=y_val)
        return {'loss': loss, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(f_nn, space_, algo=tpe.suggest, max_evals=50,trials=trials)
    print('best: ' + str(best))


def to_int(params, key, *args):
    for arg in args:
        if arg not in params:
            return
        params = params[arg]
    if key in params:
        params[key] = int(params[key])
    else:
        return


hypermin(space, create_model, sin_input, sin_output, sin_input, sin_output,
         verbose=0,
         callbacks=[keras.callbacks.TensorBoard()])
