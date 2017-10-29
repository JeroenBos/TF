import types
import keras
import os

_modules = [keras.optimizers, keras.activations, keras.losses]


def get_name(parameter):
    """ Gets the name of the parameter, by mapping the function or type back to its name. """
    if isinstance(parameter, str):
        return '\'' + str(parameter) + '\''
    elif isinstance(parameter, (types.FunctionType, type)):
        type_or_module = next(m for m in _modules if hasattr(m, parameter.__name__))
        return type_or_module.__name__ + '.' + parameter.__name__
    elif 'object at 0x' not in str(parameter):
        return str(parameter)
    else:
        return type(parameter).__name__


def try_find(params, directory):
    path = os.path.join(directory, get_filename(params))
    return keras.models.load_model(path) if os.path.exists(path) else None


def get_filename(params):
    return ",".join(_get_filenamepart(params)) + '.hdf5'


def _get_filenamepart(params):
    for (key, param) in params.items():
        if isinstance(param, dict):
            for part in _get_filenamepart(param):
                yield part
        else:
            yield str(key) + '=' + get_name(param)


class Save(keras.callbacks.Callback):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory

    def on_train_end(self, logs=None):
        self.save()

    def save(self):
        path = os.path.join(self.directory, get_filename(self.model.parameters))
        self.model.save(path)
