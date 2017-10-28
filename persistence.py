import types
import keras
import datetime
import os

_modules = [keras.optimizers, keras.activations, keras.losses]


def get_name(parameter):
    """ Gets the name of the parameter, by mapping the function or type back to its name. """
    if isinstance(parameter, (types.FunctionType, type)):
        type_or_module = next(m for m in _modules if hasattr(m, parameter.__name__))
        return type_or_module.__name__ + '.' + parameter.__name__
    else:
        return str(parameter)


def try_find(params, directory):
    path = os.path.join(directory, get_filename(params))
    return keras.models.load_model(path) if os.path.exists(path) else None


def get_filename(params):
    return "a.hdf5"  # return datetime.datetime.now().strftime('%H:%M:%S') + '.hdf5'


class Save(keras.callbacks.Callback):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory

    def on_train_end(self, logs=None):
        self.save()

    def save(self):
        path = os.path.join(self.directory, get_filename(self.params)) # TODO: I don't know what these params are actually
        self.model.save(path)


