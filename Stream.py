from zipfile import ZipFile
import fileinput as fi
from queue import Queue
import concurrent.futures
from threading import current_thread
"""
    Модуль отвечающий за поставку данных
"""


def _open_hook(filename, mode):
    """
            https://docs.python.org/3/library/fileinput.html#fileinput.FileInput
       :param filename:
       :param mode:
       :return: opened .csv or .txt fileobject or opened fileobject in .zip archive
    """
    if filename.endswith(('.csv', '.txt')):
        return open(filename, mode)
    if filename.endswith('.zip'):
        try:
            zf = ZipFile(filename)
            return zf.open(filename.replace('.zip', '.csv'), mode=mode)
        except Exception as e:
            raise e


def fileStream(files, batch_size=32, *args, **kwargs):
    """
     Generates batches from file lines (expected float number per line)
    :param files: iterable of filenames
    :param batch_size: batch size
    :return: batches of data generator
    """
    eof = False
    with fi.input(files=files, mode='r', openhook=_open_hook) as f:
        while not eof:
            r = []
            for i in range(batch_size):
                line = f.readline()
                if line:
                    try:
                        r.append(float(line))
                    except ValueError as e:
                        print(e, "Couldn't read float")
                else:
                    eof = True
                    break
            yield r


class Stream:

    def __init__(self,max_size=0):
        self.queue = Queue(maxsize=max_size)

    def pop(self):
        pass

    def put(self):
        pass


class FrameStream(Stream):

    def __init__(self, max_size=0, *args,**kwargs):
        super().__init__(max_size)

    def pop(self, batch_size=1):
        r = []
        for i in range(batch_size):
            r.append(self.queue.pop())
        return r

    def put(self, value):
        self.queue.put(value)


class FileStream(Stream):

    @staticmethod
    def _open_hook(filename, mode):
        """
                https://docs.python.org/3/library/fileinput.html#fileinput.FileInput
           :param filename:
           :param mode:
           :return: opened .csv or .txt fileobject or opened fileobject in .zip archive
        """
        if filename.endswith(('.csv', '.txt')):
            return open(filename, mode)
        if filename.endswith('.zip'):
            try:
                zf = ZipFile(filename)
                return zf.open(filename.replace('.zip', '.csv'), mode=mode)
            except Exception as e:
                raise e

    def __init__(self, files, max_size = 0,*args,**kwargs):

        super().__init__(max_size)
        self.f = fi.input(files=files, mode=kwargs.get('mode', 'r'), openhook=self._open_hook)
        self.eof = False

    def pop(self, batch_size=32, *args, **kwargs):
        """
         Generates batches from file lines (expected float number per line)
        :param files: iterable of filenames
        :param batch_size: batch size
        :return: batches of data generator
        """
        r = []
        for i in range(batch_size):
            r.append(self.queue.get())
        return r

    def load_from_files(self):
        print('Hello! thread:',current_thread())
        while True:
            line = self.f.readline()
            if line:
                try:
                    self.queue.put(float(line))
                except ValueError as e:
                    print(e, "Couldn't interpret line as float")
            else:
                self.eof=True
                break

    def put(self):
        pass

    def __del__(self):
        self.f.close()


