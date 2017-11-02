import threading
import queue


class MyQueue:
    __init_lock = threading.RLock()
    """
    Assumes there is only one reading thread and possible multiple writing threads
    """
    def __init__(self):
        self.__getter = None
        self.__get_i = 0
        self.__get_data = []
        self.__put_data = {}
        self.__lock = None

    def init(self):
        if self.__lock:
            return

        acquired = MyQueue.__init_lock.acquire(blocking=False)
        if acquired:
            self.__lock = threading.RLock()
            MyQueue.__init_lock.release()
        else:
            while not self.__lock:  # wait for other thread with init lock to take ownership of this instance
                pass

    def put(self, item):
        if not self.__lock:
            raise ValueError('forgot init')
        if threading.get_ident() not in self.__put_data:
            with self.__lock:
                self.__put_data[threading.get_ident()] = []

        self.__put_data[threading.get_ident()].append(item)

    def get(self):
        if not self.__lock:
            raise ValueError('forgot init')
        if not self.__getter:
            with self.__lock:
                if self.__getter:
                    raise ValueError('Multiple getters')
                self.__getter = threading.get_ident()
        if self.__getter != threading.get_ident():
            raise ValueError('Multiple getters')

        if self.__get_i == len(self.__get_data):
            self.__get_i = 0
            self.__get_data.clear()
            with self.__lock:
                for key, new_data in self.__put_data:
                    self.__get_data += new_data
                    self.__put_data[key].clear()

        if self.__get_i == len(self.__get_data):
            raise queue.Empty

        result = self.__get_data[self.__get_i]
        self.__get_i += 1
        return result

    def qsize(self):
        """Not exact"""
        if not self.__lock:
            raise ValueError('forgot init')

        result = len(self.__get_data) - self.__get_i
        for key in self.__put_data:
            try:
                result += len(self.__put_data[key])
            except KeyError:
                pass
        return result







