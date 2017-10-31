import multiprocessing
import queue
import matplotlib.pyplot as plt

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
        plt.pause(refresh_rate)


def plot(worker, **kwargs):
    assert callable(worker)
    enqueue(_Wrapper(worker).do, **kwargs)


class _Wrapper:  # this class makes the function 'do' pickable
    def __init__(self, worker):
        self.__worker = worker

    def do(self, **kwargs):
        plt.ion()
        plt.show()
        plt.clf()

        self.__worker(**kwargs)

        plt.draw()


