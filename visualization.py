import threading
import collections
import multiprocessing
import queue
from datetime import datetime
import time
import matplotlib.pyplot as plt


q = None
refresh_rate = 0.01  # in seconds
_NO_KEY = object()
axes = None


def enqueue(worker, worker_key=None, **kwargs):
    worker(**kwargs)
    return
    assert worker
    assert callable(worker)
    global q
    if q is None:
        q = multiprocessing.Queue()
        process = multiprocessing.Process(target=_worker, args=(q,))
        process.start()
    q.put((worker, worker_key, kwargs))
    print('putted item:', kwargs['gen'])


# noinspection PyArgumentList
def _worker(local_queue):
    worker_count = 0
    performed_tasks_index = 0
    tasks = {}
    while True:
        try:  # only do newest enqueued element per key (no need in plotting old results)
            while True:
                worker, key, kwargs = local_queue.get(block=False)
                time.sleep(0)
                print('gotten item:', kwargs['gen'])
                key = key or _NO_KEY
                if key in tasks and tasks[key][0]:
                    tasks[key] = worker, kwargs, tasks[key][2]
                else:
                    if key not in tasks:
                        worker_count += 1
                        print('new worker count: ' + str(worker_count))
                    tasks[key] = worker, kwargs, performed_tasks_index
                    performed_tasks_index += 1

        except queue.Empty:
            while True:
                key, (worker, kwargs, _prio) = max(filter(lambda kvp: kvp[1][2] is not None, tasks.items()),
                                                   key=lambda kvp: kvp[1][2],
                                                   default=(None, (None, None, None)))

                if key and worker:
                    tasks[key] = None, None, None
                    # noinspection PyCallingNonCallable
                    worker(**kwargs)
                else:
                    break
        plt.pause(refresh_rate)


def plot(worker, subplot_index=(1, 1), **kwargs):
    assert callable(worker)
    enqueue(_Wrapper(worker, subplot_index).do, subplot_index, **kwargs)


def _get_max_index(subplotters):
    ncols = max(subplotters, key=lambda kvp: kvp[0])[0]
    nrows = max(subplotters, key=lambda kvp: kvp[1])[1]
    return ncols, nrows

def _ensure_axes(subplot_index):
    global axes
    if axes is None:
        axes = {}
        plt.ion()
        _, a = plt.subplots(subplot_index[1], subplot_index[0], squeeze=False)
        for x in range(subplot_index[0]):
            for y in range(subplot_index[1]):
                axes[(x + 1, y + 1)] = a[y, x]

    if subplot_index not in axes:
        ncols = max(axes, key=lambda kvp: kvp[0])[0]
        nrows = max(axes, key=lambda kvp: kvp[1])[1]
        i = subplot_index[0] + subplot_index[1] * ncols
        fig = plt.gcf()
        assert fig
        axes[subplot_index] = fig.add_subplot(i, ncols, nrows)

    return axes[subplot_index]


class _Wrapper:  # this class makes the function 'do' pickable
    def __init__(self, worker, subplot_index):
        self.__worker = worker
        self.__subplot_index = subplot_index

    def do(self, **kwargs):
        subplot = _ensure_axes(self.__subplot_index)
        subplot.clear()

        self.__worker(subplot=subplot, **kwargs)

        fig = plt.get_current_fig_manager()
        fig.canvas.draw()
        plt.pause(refresh_rate)


def loop(q, subplotters):
    global axes
    _ensure_axes(_get_max_index(subplotters))
    t = None
    while True:
        try:
            time.sleep(0.1)
            while True:  # only get latest
                t = q.get(block=False)
        except queue.Empty:
            if t:
                for subplot_index, plotter in subplotters.items():
                    axes[subplot_index].clear()
                    plotter(axes[subplot_index], *t)
                fig = plt.get_current_fig_manager()
                fig.canvas.draw()
        finally:
            plt.pause(refresh_rate)
