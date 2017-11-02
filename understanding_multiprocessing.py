import threading
import multiprocessing
import queue
import time

q = None


def enqueue(item):
    global q
    if q is None:
        q = multiprocessing.Queue()
        process = threading.Thread(target=worker, args=(q,))
        process.start()
    q.put(item)
    print('putted item:', item)
    #time.sleep(0)


def worker(local_queue):
    while True:
        try:  # only do newest enqueued element per key (no need in plotting old results)
            while True:
                item = local_queue.get(block=False)
                print('gotten item: ', item)
        except queue.Empty:
            pass #    print('empty')
        #time.sleep(1)


if __name__ == '__main__':
    for i in range(200000):
        enqueue(i)
        #for j in range(1000000):
        #    j = j + 1

