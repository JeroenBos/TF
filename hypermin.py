from hyperopt import fmin, tpe, STATUS_OK, Trials


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
