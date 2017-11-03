import threading
import multiprocessing
import queue

q = None


def enqueue(item):
    global q
    if q is None:
        q = multiprocessing.Queue()
        process = threading.Thread(target=worker, args=(q,))
        process.start()
    q.put(item)
    print(f'len putted item: {len(item)}. qsize: {q.qsize()}. buffer len: {len(q._buffer)}')


def worker(local_queue):
    while True:
        try:
            while True:
                item = local_queue.get(block=False)
                print(f'len got item: {len(item)}. qsize: {q.qsize()}. buffer len: {len(q._buffer)}')
        except queue.Empty:
            print('empty')


if __name__ == '__main__':
    for i in range(1, 100000, 1000):
        enqueue(list(range(i)))


