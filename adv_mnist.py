from math import sin, pi
from itertools import islice
import tensorflow as tf
import numpy as np
import keras.initializers
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Conv2D, Reshape, UpSampling2D, Flatten
from keras import losses, activations, callbacks, optimizers
import persistence
from visualization_callbacks import OneDValidationContinuousPlotCallback
from persistence import *
import mnist
import numpy as np
from functools import reduce
import operator
import random
import sys
from math import log10, floor


def debugging():
    return sys.gettrace()


LOG_DIRECTORY = "D:\\TFlogs\\"
NOISE_SIZE = 100
BATCH_SIZE = 10 if debugging() else 1000

space = {'units2': 40,
         'units3': 28*28,
         'epochs': 2
         }

def get_mnist_x():
    x_train, y_train, x_val, y_val, x_test, y_test = mnist.load_dataset()
    data = np.concatenate((x_train, x_val, x_test))
    return data


def load_model():
    model = keras.models.load_model(persistence.get_filename(mnist.space))
    model.trainable = False
    return model


def size_equals(shape1, shape2):
    def size(shape):
        return reduce(operator.mul, shape, 1)
    return size(shape1) == size(shape2)


def Dchoice(a, size):
    indices = np.random.randint(len(a), size=size)
    result = list(a[i] for i in indices)
    return np.array(result)


def create_model(params, input_shape):
    persistence.print_param_names(params)

    generator = Sequential()
    generator.add(Dense(NOISE_SIZE, input_shape=(NOISE_SIZE,)))
    generator.add(Dense(units=params['units2'], activation=activations.tanh))
    generator.add(Dense(units=params['units3'], activation=activations.sigmoid))

    assert size_equals(generator.output_shape[1:], input_shape[1:])
    generator.add(Reshape(input_shape[1::]))

    assert generator.output_shape[1:] == input_shape[1:]

    discriminator = Sequential()
    discriminator.add(Conv2D(filters=32,
                             kernel_size=3,
                             activation=activations.relu,
                             input_shape=input_shape[1:]))
    discriminator.add(Flatten())
    discriminator.add(Dense(2, activation=activations.softmax))
    discriminator.compile(loss=losses.categorical_crossentropy,
                          optimizer=optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8))

    print('discriminator output shape:', discriminator.output_shape)
    both = Sequential()
    both.add(generator)
    both.add(discriminator)
    discriminator.trainable = False
    both.compile(loss=losses.categorical_crossentropy,
                 optimizer=optimizers.SGD(lr=0.1),
                 metrics=['accuracy'])

    return generator, discriminator, both


def train_adversarial(models, real_data, batch_size, verbose=0):
    generator, discriminator, both = models
    y_fakes = keras.utils.to_categorical(np.zeros(batch_size), 2)
    y_reals = keras.utils.to_categorical(np.ones (batch_size), 2)
    y = np.concatenate((y_reals, y_fakes))

    while True:
        noises = np.array(list(np.random.uniform(0, 1, NOISE_SIZE) for _ in range(batch_size)))
        fakes = generator.predict(noises)
        x = np.concatenate((Dchoice(real_data, batch_size), fakes))

        discriminator.trainable = True
        d_metrics = discriminator.fit(x, y, epochs=1, verbose=verbose)
        discriminator.trainable = False
        b_metrics = both.fit(noises, y_reals, epochs=10, verbose=verbose)

        # print(discriminator.predict(x))
        yield d_metrics, b_metrics


def round_to_2_significant(x):
    return round(x, 1-int(floor(log10(abs(x)))))


if __name__ == '__main__':
    data = get_mnist_x()

    def summarize(metrics):
        return round_to_2_significant(metrics.history[loss][-1])

    for d_metric, b_metric in islice(train_adversarial(create_model(space, data.shape), data, BATCH_SIZE), 0, 10000):
        loss = 'loss'

        print(f'disc loss: {summarize(d_metric):f}. gen loss: {summarize(b_metric):f}')



