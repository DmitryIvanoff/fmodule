from zipfile import ZipFile
import fileinput as fi
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


def fileStream(self, files, batch_size=32, *args, **kwargs):
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

    def __init__(self):
        pass

    def get_values(self):
        pass


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

    def __init__(self, files,*args,**kwargs):

        super().__init__()
        self.f = fi.input(files=files, mode=kwargs.get('mode', 'r'), openhook=self._open_hook)
        self.eof = False

    def get_values(self, files, batch_size=32, *args, **kwargs):
        """
         Generates batches from file lines (expected float number per line)
        :param files: iterable of filenames
        :param batch_size: batch size
        :return: batches of data generator
        """
        r = []
        if not self.eof:
            for i in range(batch_size):
                line = self.f.readline()
                if line:
                    try:
                        r.append(float(line))
                    except ValueError as e:
                        print(e, "Couldn't read float")
                else:
                    self.eof = True
                    break
        else:
            pass
            #wait till lines are added to file
        return r

    def __del__(self):
        self.f.close()
