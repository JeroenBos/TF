import multiprocessing
import queue
import matplotlib.pyplot as plt

q = None
refresh_rate = 0.5  # in seconds
_NO_KEY = object()


def enqueue(worker, worker_key=None, **kwargs):
    assert worker
    assert callable(worker)
    global q
    if q is None:
        q = multiprocessing.JoinableQueue()
        process = multiprocessing.Process(target=_worker, args=(q,))
        process.start()
    q.put((worker, worker_key, kwargs))


def _worker(local_queue):
    worker_count = 0
    performed_tasks_index = 0
    tasks = {}

    while True:
        try:  # only do newest enqueued element per key (no need in plotting old results)
            while True:
                worker, key, kwargs = local_queue.get(False)
                key = key or _NO_KEY
                if key in tasks and tasks[key]:
                    local_queue.task_done()
                    tasks[key] = worker, kwargs, tasks[key][2]
                else:
                    if key not in tasks:
                        worker_count += 1
                        print('new worker count: ' + str(worker_count))
                    tasks[key] = worker, kwargs, performed_tasks_index
                    performed_tasks_index += 1

        except queue.Empty:
            key, (worker, kwargs, _prio) = max(filter(lambda kvp: kvp[1][2], tasks.items()),
                                               key=lambda kvp: kvp[1][2],
                                               default=(None, (None, None, None)))
            if key and worker:
                tasks[key] = None, None, None
                worker(**kwargs)
                local_queue.task_done()

        # on empty or non-empty queue: yield control back to the plot
        plt.pause(refresh_rate)


def plot(worker, subplot_index=(1, 1), **kwargs):
    assert callable(worker)
    enqueue(_Wrapper(worker, subplot_index).do, subplot_index, **kwargs)


class _Wrapper:  # this class makes the function 'do' pickable
    def __init__(self, worker, subplot_index):
        self.__worker = worker
        self.__subplot_index = subplot_index  # TODO use somewhere or delete

    def do(self, **kwargs):
        plt.ion()
        plt.show()
        plt.clf()

        self.__worker(**kwargs)

        plt.draw()
