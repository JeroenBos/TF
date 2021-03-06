import sys
from hyperopt import fmin, tpe, STATUS_OK, Trials
import keras


def hypermin(space, to_model, x, y, x_val, y_val, **kwargs):
    """Maps the hyperspace search loss to the model.fit loss for regression output"""

    def f_nn(params):
        _flatten(params)
        model = to_model(params, input_shape=x.shape[1:])

        class SetLossCallBack(keras.callbacks.Callback):
            def on_train_end(self, logs=None):
                self.model.hploss = self.model.evaluate(x=x_val, y=y_val)

        kwargs['callbacks'] = [SetLossCallBack()] + (kwargs['callbacks'] or [])
        model.fit(x, y, epochs=params['epochs'], **kwargs)

        return {'loss': model.hploss[0] if isinstance(model.hploss, list) else model.hploss, 'status': STATUS_OK}

    trials = Trials()
    fmin(f_nn, space, algo=tpe.suggest, max_evals=sys.maxsize, trials=trials)


def _flatten(params):
    subdicts = [(key, param) for key, param in params.items() if isinstance(param, dict)]
    for (key, subdict) in subdicts:
        del params[key]
        for subkey, value in subdict.items():
            params[subkey] = value

