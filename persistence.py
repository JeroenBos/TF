import types
import keras
import os
import tensorflow as tf

# list of modules from which try to find the names of parameters (for file names when saving/loading)
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
    """Tries to find a saved model in the specified directory that matches the specified parameters"""
    path = os.path.join(directory, get_filename(params))
    return keras.models.load_model(path) if os.path.exists(path) else None


def get_filename(params):
    """Gets the canonical name for the file name when saving the model constructed from the specified parameters"""
    return ",".join(_get_filenamepart(params)) + '.hdf5'


def _get_filenamepart(params):
    for (key, param) in params.items():
        if isinstance(param, dict):
            for part in _get_filenamepart(param):
                yield part
        else:
            yield str(key) + '=' + get_name(param)


def print_param_names(params):
    """Prints the specified parameters to the console"""
    print()
    for line in sorted(_get_filenamepart(params)):
        print(line)


class Save(keras.callbacks.Callback):
    """A callback that saves the model at the end of its training"""
    def __init__(self, directory):
        super().__init__()
        self.directory = directory

    def on_train_end(self, logs=None):
        self.save()

    def save(self):
        path = os.path.join(self.directory, get_filename(self.model.parameters))
        self.model.save(path)


class TensorBoardSummaryScalars(keras.callbacks.Callback):
    def __init__(self, log_dir, scalars):
        """scalars: a dict of strings(tags) and functions taking a model and returning a tensor """
        super().__init__()
        self.log_dir = log_dir
        self.scalars = scalars

    def on_train_begin(self, logs=None):
        for tag, get_tensor in self.scalars.items():
            tensor = get_tensor(self.model)
            self.model.metrics_tensors.append(tensor)
            self.model.metrics_names.append(tag)

    def on_epoch_end(self, epoch, logs=None):
        assert logs
        sess = keras.backend.get_session()
        for tag, get_tensor in self.scalars.items():
            tensor = get_tensor(self.model)
            val = tensor.eval(sess)
            logs[tag] = val


