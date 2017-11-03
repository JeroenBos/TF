import matplotlib.pyplot as plt
import keras
import visualization


class PlotCallback(keras.callbacks.Callback):
    def __init__(self, plot_worker, get_plot_worker_args, select_new_model_predicate=None):
        super().__init__()
        self.__select_new_model_predicate = select_new_model_predicate or (lambda new, best: new.hploss < best.hploss)
        self.selected_model = None
        self.__plot_worker = plot_worker
        self.__get_plot_worker_args = get_plot_worker_args

    def on_train_end(self, logs=None):
        self.update()

    def update(self):
        if not self.selected_model or self.__select_new_model_predicate(self.model, self.selected_model):
            self.selected_model = self.model

            # omg this is so ugly, but has to be done because of pickling when going between threads. Aaargh
            visualization.plot(self.__plot_worker, **self.__get_plot_worker_args())


class OneDValidationPlotCallback(PlotCallback):
    def __init__(self, x_val, y_val, select_new_model_predicate=None, x_map=None):
        """select_new_model_predicate: a function taking the current model. Returns whether it pertains to the plot"""
        """x_map: a function taking some input x, and mapping it the x-axis. """
        self.__x_val = x_val
        self.__y_val = y_val
        self.__x_map = x_map or (lambda x_val_: x_val_)
        super().__init__(self.plot, self.get_plot_args, select_new_model_predicate)

    def get_plot_args(self):
        x_axis_values = self.__x_map(self.__x_val)
        return {'x_val': x_axis_values,
                'y_val': self.__y_val,
                'predictions': self.selected_model.predict(self.__x_val)}

    @staticmethod
    def plot(x_val, y_val, predictions):
        plt.scatter(x_val, predictions)
        plt.scatter(x_val, y_val)


class OneDValidationContinuousPlotCallback(OneDValidationPlotCallback):
    def __init__(self, x_val, y_val, x_map=None):

        super().__init__(x_val, y_val, lambda *args: True, x_map)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            self.update()


class PrintEpoch(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.__epoch = 0