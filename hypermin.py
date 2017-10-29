import sys
from hyperopt import fmin, tpe, STATUS_OK, Trials


def hypermin(space_, to_model, x, y, x_val, y_val, **kwargs):
    print("x.shape=" + str(x.shape))

    def f_nn(params):
        _flatten(params)
        model = to_model(params, input_dim=x.shape[1])
        model.fit(x, y, epochs=params['epochs'], **kwargs)

        loss = model.evaluate(x=x_val, y=y_val)
        return {'loss': loss, 'status': STATUS_OK}

    trials = Trials()
    fmin(f_nn, space_, algo=tpe.suggest, max_evals=sys.maxsize, trials=trials)


def _flatten(params):
    subdicts = [(key, param) for key, param in params.items() if isinstance(param, dict)]
    for (key, subdict) in subdicts:
        del params[key]
        for subkey, value in subdict.items():
            params[subkey] = value



