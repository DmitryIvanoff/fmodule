from bitarray import bitarray
import numpy as np
import random as rand
from secrets import token_hex
import time as Clock

def buftoint(ba):
    """
       Func is intended to transform bitarray in int value
        - throws ValueError
    :param ba: bitarray
    :return: int
    """
    r = None
    if ba:
        r=int.from_bytes(ba.tobytes(), ba.endian())
    else:
        raise ValueError("set valid buffer")
    return r


class Histogram:
    """
        Histogram gather information about time intervals(offsets)
        between same slicing buffers (with same int value) appearance in binary series
    """
    def __init__(self, max_offset, bufflength, dtype=np.dtype('uint32'), *args, **kwargs):
        '''
        :param max_offset: max time interval between same buffers
        :param bufflength: slicing buffer length
        :param dtype: numpy.dtype of histogram and offsets
        :param args:
        :param kwargs: endian ('big' or 'little')
        '''
        self.hist = np.zeros((2<<bufflength-1, max_offset), dtype=dtype)
        self.offset = np.zeros(2<<bufflength-1, dtype=dtype)
        self.slbuffer = bitarray(bufflength, endian=kwargs.get('endian', 'big'))
        self.slbuffer.setall(False)
        self.max_offset = max_offset
        self.prediction=False
        self.accuracy = 0
        self.timestamp = 0
        self.sum = 0
        rand.seed(kwargs.get('seed', token_hex(16)))

    def take_probabilities(self):
        '''
             Predicts next value of binary series
             for 1 time step
        :return: prediction (True (1), False(0))
        '''
        buff = self.slbuffer[1:]

        # calc probability of 0

        buff.append(False)

        index = buftoint(buff)
        # get offset
        m, time_index = divmod(self.offset[index], self.max_offset)
        if not m:
            prob_0 = self.hist[index, time_index]
        else:
            prob_0 = 0

        # calc probability of 1

        index += 1
        # get offset
        m, time_index = divmod(self.offset[index], self.max_offset)
        if not m:
            prob_1 = self.hist[index, time_index]
        else:
            prob_1 = 0

        if self.sum > 0:
            norm_multiplier = 1/self.sum
        else:
            norm_multiplier = 0.0

        # make prediction
        if prob_1 != prob_0:
            self.prediction = prob_1 > prob_0
        else:
            self.prediction = rand.randint(0, 1)
        return (prob_0*norm_multiplier), (prob_1*norm_multiplier)

    def step(self, series):
        """
           Moves slicing buffer for 1 time step forward
           and updates the Histogram and offsets

        :param series: new value of binary series (True or False)
        """
        self.timestamp += 1

        # move buffer
        self.slbuffer.pop(0)
        self.slbuffer.append(series)

        val_index = buftoint(self.slbuffer)
        m, time_index = divmod(self.offset[val_index], self.max_offset)

        if not (self.prediction ^ series):
            self.accuracy += 1

        # update histogram
        if not m:
            self.hist[val_index, time_index] += 1
            self.sum += 1

        # update offsets
        self.offset[val_index] = 0
        self.offset += 1

