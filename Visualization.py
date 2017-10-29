import multiprocessing
import queue
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import keras
import numpy as np
from PushFuncAnimation import PushFuncAnimation

q = None
refresh_rate = 0.5  # in seconds


def enqueue(worker, **kwargs):
    global q
    if q is None:
        q = multiprocessing.JoinableQueue()
        process = multiprocessing.Process(target=_worker, args=(q,))
        process.start()
    q.put((worker, kwargs))


def _worker(local_queue):
    while True:
        worker = None
        kwargs = None
        try:  # only do newest enqueued element (no need in plotting old results)
            worker, kwargs = local_queue.get(False)
            while True:
                worker, kwargs = local_queue.get(False)
                local_queue.task_done()
        except queue.Empty:
            if worker is not None:
                worker(**kwargs)
                local_queue.task_done()
        # on empty or non-empty queue: yield control back to the plot
        # plt.pause(refresh_rate)


def plot(worker, **kwargs):
    assert callable(worker)
    enqueue(_Wrapper(worker).update_artists, **kwargs)


class _Wrapper:  # this class makes the function 'do' pickable
    def __init__(self, worker):
        self.__worker = worker
        self.__ani = None

    def update_artists(self, **kwargs):
        new_artists = self.__worker(**kwargs)
        if not new_artists:
            raise Exception("artists must be returned")

        if not self.__ani:  # i.e. these are the first artists
            self.__ani = self.init_animation()

        self.__ani.update(new_artists)

    def init_animation(self):
        fig, ax = plt.subplots()
        ani = PushFuncAnimation(fig, self.__worker)
        plt.show()
        return ani



class PlotCallback(keras.callbacks.Callback):
    def __init__(self, plot_worker, get_plot_worker_args, select_new_model_predicate=None):
        super().__init__()
        self.__select_new_model_predicate = select_new_model_predicate or (lambda tested, best: tested.hploss < best.hploss)
        self.selected_model = None
        self.__plot_worker = plot_worker
        self.__get_plot_worker_args = get_plot_worker_args

    def on_train_end(self, logs=None):
        self.update()

    def update(self):
        if not self.selected_model or self.__select_new_model_predicate(self.model, self.selected_model):
            self.selected_model = self.model

            # omg this is so ugly, but has to be done because of pickling when going between threads. Aaargh
            plot(self.__plot_worker, **self.__get_plot_worker_args())


class OneDValidationPlotCallback(PlotCallback):
    def __init__(self, x_val, y_val, select_new_model_predicate=None):
        self.__x_val = x_val
        self.__y_val = y_val
        self.__scat1 = None
        self.__scat2 = None
        super().__init__(self.plot, self.get_plot_args, select_new_model_predicate)

    def get_plot_args(self):
        if not self.__scat1:
            self.init_scats()

        return {'x_val': self.__x_val,
                'y_val': self.__y_val,
                'predictions': self.selected_model.predict(self.__x_val),
                'scat1': self.__scat1,
                'scat2': self.__scat2 }


    def init_scats(self):
        self.__scat1 = plt.scatter(self.__x_val, self.selected_model.predict(self.__x_val))
        self.__scat2 = plt.scatter(self.__x_val, self.__y_val)
        return (self.__scat1, self.__scat2)


    @staticmethod
    def plot(scat1, scat2, x_val, y_val, predictions):
        scat1.set_offsets(np.c_[x_val, predictions])
        scat2.set_offsets(np.c_[x_val, y_val])
        return (scat1, scat2)


class OneDValidationContinuousPlotCallback(OneDValidationPlotCallback):
    def __init__(self, x_val, y_val):
        super().__init__(x_val, y_val, lambda *args: True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            self.update()
