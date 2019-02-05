from bitarray import bitarray
import numpy as np
import random as rand
from secrets import token_hex

def buftoint(ba):
    r=None
    if ba:
        r=int.from_bytes(ba.tobytes(), ba.endian())
    else:
        raise ValueError("set valid buffer")
    return r


class Histogram:
    """

    """
    def __init__(self, num_series, max_offset, bufflength, dtype=np.dtype('uint32'), *args, **kwargs):
        '''

        :param num_series:
        :param max_offset:
        :param bufflength:
        :param dtype:
        :param args:
        :param kwargs:
        '''
        self.hist = np.zeros((num_series, 2<<bufflength-1, max_offset), dtype=dtype)
        self.offset = np.zeros((num_series, 2<<bufflength-1), dtype=dtype)
        init_buffer = bitarray(bufflength, endian=kwargs.get('endian', 'big'))
        init_buffer.setall(False)
        self.slbuffers = [init_buffer for i in range(num_series)]
        self.num_series = num_series
        self.max_offset = max_offset
        self.accuracy = np.zeros(num_series, dtype=np.uint32)
        self.timestamp = 0
        self.prediction = [False for i in range(self.num_series)]
        rand.seed(kwargs.get('seed', token_hex(16)))

    def step(self, series):
        self.timestamp += 1
        if len(series) != self.num_series:
            raise RuntimeError('length of given series tuple doesn\'t equals num_series')
        else:
            for i in range(len(series)):
                # move buffer
                self.slbuffers[i].pop(0)
                buff = bitarray(self.slbuffers[i])
                self.slbuffers[i].append(series[i])

                # calc probability of 0
                buff.append(False)
                index = buftoint(buff)
                m, time_index = divmod(self.offset[i, index], self.max_offset)
                if not m:
                    probability_0 = self.hist[i, index, time_index]
                else:
                    probability_0 = 0
                buff.pop()

                # calc probability of 1

                buff.append(True)
                # get buff as int
                index = buftoint(buff)
                # get offset
                m, time_index = divmod(self.offset[i, index], self.max_offset)
                if not m:
                    probability_1 = self.hist[i, index, time_index]
                else:
                    probability_1 = 0

                # make prediction
                if probability_1 != probability_0:
                    self.prediction[i] = (probability_1 > probability_0)
                else:
                    self.prediction[i] = rand.randint(0, 1)

                # calc accuracy
                if not (self.prediction[i] ^ series[i]):
                    self.accuracy[i] += 1

                val_index = buftoint(self.slbuffers[i])
                m, time_index = divmod(self.offset[i][val_index], self.max_offset)

                # update histogram
                if not m:
                    self.hist[i, val_index, time_index] += 1

                # update offsets
                self.offset[i, val_index] = 0
                self.offset[i] += 1

            return self.prediction
