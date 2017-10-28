import types
import keras

_modules = [keras.optimizers, keras.activations, keras.losses]


def get_name(parameter):
    """ Gets the name of the parameter, by mapping the function or type back to its name. """
    if isinstance(parameter, (types.FunctionType, type)):
        type_or_module = next(m for m in _modules if hasattr(m, parameter.__name__))
        return type_or_module.__name__ + '.' + parameter.__name__
    else:
        return str(parameter)
