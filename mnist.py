import os
import numpy as np
import gzip
import pickle
from urllib import request
import persistence
from visualization_callbacks import OneDValidationContinuousPlotCallback
from persistence import *
from keras.layers import *
from keras import activations

DIRECTORY = "D:\\"

def load_dataset():
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    filename = 'mnist.pkl.gz'
    if not os.path.exists(filename):
        print("Downloading MNIST dataset...")
        request.urlretrieve(url, filename)
        print("Downloaded MNIST dataset...")
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    x_train, y_train = data[0]
    x_val, y_val = data[1]
    x_test, y_test = data[2]
    shape = (-1, 28, 28, 1)
    x_train = x_train.reshape(shape)
    x_val = x_val.reshape(shape)
    x_test = x_test.reshape(shape)
    y_train = keras.utils.to_categorical(y_train.astype(np.uint8), 10)
    y_val = keras.utils.to_categorical(y_val.astype(np.uint8), 10)
    y_test = keras.utils.to_categorical(y_test.astype(np.uint8), 10)
    return x_train, y_train, x_val, y_val, x_test, y_test


space = {'units1': 40,
         'units2': 40,
         'units3': 10,
         'learning_rate': 0.2,
         'loss': keras.losses.categorical_crossentropy,
         'epochs': 2
         }


def create_model(params, input_shape):
    persistence.print_param_names(params)

    model = keras.models.Sequential()
    model.parameters = params # for callbacks.Save
    model.add(Conv2D(filters=32,
                     kernel_size=3,
                     activation=activations.relu,
                     input_shape=input_shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, 3, activation=activations.relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation=activations.relu))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation=activations.softmax))
    model.compile(optimizer=keras.optimizers.SGD(params['learning_rate']), loss=params['loss'], metrics=['accuracy'])
    return model


if __name__ == '__main__':
    x, y, x_val, y_val, _, _ = load_dataset()
    model = create_model(space, x.shape)

    model.fit(x, y, validation_data=(x_val, y_val), epochs=space['epochs'], verbose=1, callbacks=[Save(DIRECTORY)])



# note: don't know where the 50000 is coming from