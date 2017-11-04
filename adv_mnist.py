import multiprocessing
import matplotlib.pyplot as plt
from math import sin, pi
from itertools import islice
import tensorflow as tf
import numpy as np
import keras.initializers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Reshape, UpSampling2D, Flatten, SpatialDropout2D
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
from visualization import loop


def debugging():
    return sys.gettrace()


LOG_DIRECTORY = "D:\\TFlogs\\"
NOISE_SIZE = 100
BATCH_SIZE = 10 if debugging() else 1000

space = {'units2': 40,
         'units3': 28*28,
         'epochs': 2
         }

def load_mnist_x():
    x_train, y_train, x_val, y_val, x_test, y_test = mnist.load_dataset()
    data = np.concatenate((x_train, x_val, x_test))
    return data


def load_model():
    filename = persistence.get_filename(mnist.space)
    model = keras.models.load_model(filename)
    model.trainable = False
    return model

def size(shape):
    return reduce(operator.mul, shape, 1)

def size_equals(shape1, shape2):
    return size(shape1) == size(shape2)


def Dchoice(a, size):
    indices = np.random.randint(len(a), size=size)
    result = list(a[i] for i in indices)
    return np.array(result)


def create_model(params, input_shape):
    persistence.print_param_names(params)

    generator = Sequential()
    generator.add(Dense(NOISE_SIZE, input_shape=(NOISE_SIZE,)))
    generator.add(Dense(NOISE_SIZE))
    generator.add(Reshape((10, 10, 1)))
    generator.add(SpatialDropout2D(0.2))
    generator.add(UpSampling2D())
    generator.add(Reshape((400,)))
    generator.add(Dense(units=28*28, activation=activations.tanh))
    generator.add(Reshape((28, 28, 1)))

    assert size_equals(generator.output_shape[1:], input_shape[1:]), \
        f'{size(generator.output_shape[1:])} vs {size(input_shape[1:])}'
    generator.add(Reshape(input_shape[1:]))

    assert generator.output_shape[1:] == input_shape[1:]

    discriminator = Sequential()
    discriminator.add(Conv2D(filters=32,
                             kernel_size=3,
                             activation=activations.relu,
                             input_shape=input_shape[1:]))
    discriminator.add(Conv2D(filters=16,
                             kernel_size=3,
                             activation=activations.relu))
    discriminator.add(Flatten())
    discriminator.add(Dense(50, activation=activations.tanh))
    discriminator.add(Dense(2, activation=activations.softmax))
    discriminator.compile(loss=losses.categorical_crossentropy,
                          optimizer=optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8),
                          metrics=['accuracy'])

    print('discriminator output shape:', discriminator.output_shape)
    both = Sequential()
    both.add(generator)
    both.add(discriminator)
    discriminator.trainable = False
    both.compile(loss=losses.categorical_crossentropy,
                 optimizer=optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8),
                 metrics=['accuracy'])

    return generator, discriminator, both


class ToBeatCallback(keras.callbacks.Callback):
    def __init__(self, to_beat):
        super().__init__()
        self.to_beat = to_beat

    def on_epoch_end(self, epoch, logs=None):
        acc = logs['acc']
        if acc >= self.to_beat:
            self.model.stop_training = True


def train_adversarial(models, real_data, batch_size, callbacks=None, verbose=0):
    callbacks = callbacks if isinstance(callbacks, list) else [callbacks] if callbacks else []

    generator, discriminator, both = models

    y_fakes = keras.utils.to_categorical(np.zeros(batch_size), 2)
    y_reals = keras.utils.to_categorical(np.ones(batch_size), 2)
    y = np.concatenate((y_reals, y_fakes))

    while True:
        noises = np.array(list(np.random.uniform(0, 1, NOISE_SIZE) for _ in range(batch_size)))
        fakes = generator.predict(noises)
        x = np.concatenate((Dchoice(real_data, batch_size), fakes))

        discriminator.trainable = False
        both.fit(noises, y_reals, epochs=99999, verbose=verbose, callbacks=callbacks + [ToBeatCallback(0.5)])
        discriminator.trainable = True

        discriminator.fit(x, y, epochs=99999, verbose=verbose, callbacks=callbacks + [ToBeatCallback(0.8)])



def round_to_2_significant(x):
    if x == 0:
        return 0
    return round(x, 1-int(floor(log10(abs(x)))))

plot_label = 'acc'
def summarize(metrics):
    x = metrics.history[plot_label][-1]
    return round_to_2_significant(x)

class worker_wrapper_wrapper:
    def __init__(self, worker):
        self.worker = worker

    def __call__(self, q, kw_worker):
        for to_be_put in self.worker(**kw_worker):
            q.put(to_be_put)


def m(worker, subplotters, *data_handlers, **kw_worker):
    q = multiprocessing.Queue()
    process = multiprocessing.Process(target=worker, args=((q, kw_worker) if kw_worker else (q,)))
    process.start()

    loop(q, subplotters, *data_handlers)


class workCallback(keras.callbacks.Callback):
    # within this class, no non-constant fields on this module may be referenced because it runs on another thread
    def __init__(self, q):
        super().__init__()

        self.q = q
        data = load_mnist_x()
        train_adversarial(create_model(space, data.shape), data, BATCH_SIZE, self)

    def is_generator(self):
        return len(self.model.layers) == 2

    def on_batch_end(self, batch, logs=None):
        self.q.put((self.is_generator(), logs['loss'], logs['acc']))


if __name__ == '__main__':
    disc_x = []
    gen_x = []
    disc_loss = []
    gen_loss = []
    disc_acc = []
    gen_acc = []

    def data_handler(*args):
        gen_x.append(len(gen_x))
        disc_x.append(len(disc_x))

        if args[0]:
            loss, acc = gen_loss, gen_acc
            other_loss, other_acc = disc_loss, disc_acc
        else:
            loss, acc = disc_loss, disc_acc
            other_loss, other_acc = gen_loss, gen_acc

        loss.append(args[1])
        acc.append(args[2])

        other_loss.append(other_loss[-1] if len(other_loss) != 0 else 0)
        other_acc.append(other_acc[-1] if len(other_acc) != 0 else 0)

    def loss_plotter(subplot, *_args):
        subplot.plot(disc_x, disc_loss, label='discriminator')
        subplot.plot(gen_x, gen_loss, label='generator')
        subplot.legend(loc=(0.85, 1))
        subplot.set_ylim(bottom=0)
        subplot.set_xlim(left=0)
        subplot.set(ylabel='loss')

    def acc_plotter(subplot, *_args):
        subplot.plot(disc_x, disc_acc)
        subplot.plot(gen_x, gen_acc)
        subplot.set_ylim(bottom=0)
        subplot.set_xlim(left=0)
        subplot.set(ylabel='acc')


    m(workCallback, {(1, 1): loss_plotter, (1, 2): acc_plotter}, data_handler)




