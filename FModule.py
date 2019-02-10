"""
date: 2.02.19
Что я хочу:
Дан временной ряд. Задана история временного ряда (history_size)
Хочу с какой-то точностью предсказывать следующее значение на шаг вперед
(шаг дикретизации ряда)
Реализовать:
1. Из потока данных формировать
   a) скользящее среднее;
   б) бинарные ряды (f(t)>ma(f(t),f'(t) > 0, f''(t)>0, ...);
   в) гистограммы распределения (см. Histogram)
2. Используя гистограммы формировать прогноз, рассчитывать точность
3. Визуализация
4. Тестирование
5. Оптимизация
6. Production
"""

import numpy as np
from bitarray import bitarray
import Histogram as Hist
import Visual
from Stream import FileStream
from functools import reduce
import concurrent.futures


def ema(series, alpha, ma=None):
    """
     Calculates exponentially weighted moving average  (https://en.wikipedia.org/wiki/Moving_average)
    :param series:
    :param alpha:
    :param ma: ma(t-1)
    :return: generator(ma(t),m(t+1),..ma(t+len(series))

    """
    if not ma:
        ma = series[0]
    for f in series:
        yield alpha*f+(1.-alpha)*ma


def formBinarySeries(stream, *args, **kwargs):
    """
     (next: ma - Moving Average)
     Forms binary series of vector f,f',f'',... of
     length nod(number of derivatives)+1 like
     [ [ ma>f(t0), f'(t0)>0, f''(t0)>0, ...],
       ...................................
       [ ma>f(t), f'(t)>0, f''(t)>0, ...] ]
    :param stream: Stream of data (stream of numbers batches as for beginning)
    :return: generator of bitarray batches
    """
    alpha = kwargs.get('alpha', 0.9)  # coeff of smoothness
    nod = kwargs.get('nod', 2)  # число производных (f'(t),f''(t) as default)

    # init for ma series
    series = [bitarray()]  # series = [(ma>f(t),f'(t)>0,f''(t)>0,...] , len(series)=nod+1
    t_1 = [None]  # previous values [ma(t-1),f(t-1),v(t-1)]
    ma = None

    # init for derivatives series
    for i in range(nod):
        t_1.append(0.0)
        series.append(bitarray())

    for batch in stream:
        # print(batch)
        for i in range(nod + 1):
            series[i] = bitarray()
        for (ma, f) in zip(ema(batch, alpha=alpha, ma=t_1[0]), batch):
            series[0].append(ma > f)
            fi_dt = f
            for i in range(1, nod + 1):
                dfi_dt = fi_dt - t_1[i]
                series[i].append(dfi_dt > 0)
                t_1[i] = fi_dt
                fi_dt = dfi_dt
                # print(series)
        t_1[0] = ma
        yield series


def predict(stream, *args, **kwargs):
    """

        Gets the latest value(batch) from stream
        passes it to binarizer(formBinarySeries())
        gets binary batch and looks through Hists(time_scale)
        with various time_scales and updates hists,
        forming predictions.
        Collect all predicted values in pred list,
        forming eventual prediction array from that.
        Prediction - bool array for binary series size of
        number of derivatives (nod)

    :param stream:  iterable
    :param args:
    :param kwargs: nod,max_offset,bufflength,max_time_scale
    :return: prediction accuracy generator
    """
    # initialization of buffers and histograms
    nod = kwargs.get('nod', 2)
    max_offset = kwargs.get('max_offset', 256)
    bufflength = kwargs.get('bufflength', 8)  # bit length
    dt = np.dtype('uint32')
    max_time_scale = kwargs.get('max_time_scale', 1)
    time = 0
    hists = []
    accuracy = []
    stakes = np.zeros(nod + 1)
    prediction = [False for i in range(nod + 1)]
    for i in range(max_time_scale):
        hists.append(Hist.Histogram(nod + 1, max_offset, bufflength))
        accuracy.append([])
    print('Поехали!')
    # ------------
    for batch in formBinarySeries(stream, nod=nod):
        for series in zip(*batch):  # очередная порция данных
            time += 1
            if not (time % 100000):
                print('Живем!!', stakes / time)
                # module with T=i (quantization period) starts
            pred = []
            for i in range(1, max_time_scale + 1):
                if not (time % i):
                    pred.append(np.array(hists[i - 1].step(series)))
                    accuracy[i - 1].append(hists[i - 1].accuracy / time)
            prediction = (reduce(lambda x, y: x + y, pred) / len(pred)) > 0.5
            stakes[~prediction ^ series] += 1
        yield stakes / time


class FModule:

    def __init__(self, stream, *args, **kwargs):

        self.stream = stream
        self.nod = kwargs.get('nod', 2)
        self.max_time_scale = kwargs.get('max_time_scale', 1)
        self.max_offset = np.empty(self.nod+1, self.max_time_scale)
        self.bufflength = np.empty(self.nod+1, self.max_time_scale)

        # initialize sizes of histograms
        for i in range(self.max_time_scale):
            self.max_offset[:,i] = kwargs.get('max_offset', 256)
            self.bufflength[:,i] = kwargs.get('bufflength', 8)     # bit length

        dt = np.dtype('uint32')          # numpy.dtype of hists
        self.time = 0                    # time for accuracy control
        self.stakes = np.zeros(self.nod + 1)       # count of stakes for binary series
        self.stakes_series = (self.nod+1)*[bitarray()]  # bitarrays for stakes series

        self.hists = [[] for i in range(self.nod+1)]            # hists of various time scales
        self.accuracy = [[] for i in range(self.nod+1)]        # hists' accuracy of various time scales
        for j in range(self.nod+1):
            for i in range(self.max_time_scale):
                 self.hists[j].append(Hist.Histogram(self.max_offset[i], self.bufflength[i]))
                 self.accuracy[j].append([])

        self.alpha = kwargs.get('alpha', 0.9)         # coeff of smoothness
        self.series = [bitarray()]                    # list (nod+1 length) of bitarray batches
        self.ft_1 = [None]                            # previous values [ma(t-1),f(t-1),f'(t-1),.., f(nod-1)(t-1)]

        # init for derivatives series
        for i in range(self.nod):
            self.ft_1.append(0.0)
            self.series.append(bitarray())

    def predict(self, *args, **kwargs):
        """

            Gets the latest value(batch) from stream
            passes it to binarizer(formBinarySeries())
            gets binary batch and looks through Hists(time_scale)
            with various time_scales and updates hists,
            forming predictions.
            Collect all predicted values in pred list,
            forming eventual prediction array from that.
            Prediction - bool array for binary series size of
            number of derivatives (nod)

        :param args:
        :param kwargs: nod,max_offset,bufflength,max_time_scale
        """
        # ------------
        time_scales = kwargs.get('time_scales', tuple(range(1, self.max_time_scale + 1)))
        if min(time_scales) < 1:
            raise ValueError("Set correct time_scales!")
        prediction = [False for i in range(self.nod + 1)]

        # begin pipeline
        batch = self.stream.get_values()

        batch = self.formBinarySeries(batch)

        self.step(batch, time_scales)

        return self.stakes / self.time, self.accuracy

    def voting(self, predictions, *args,**kwargs):
        """
            Voting system for various histograms (various time_scales)
        :param predictions: predictions of every distinct histogram for every binary series
        :return: bool array of predictions for bin series
        """

        r = (np.sum(predictions, axis=1) / predictions.shape[-1]) > 0.5
        return r

    def step(self, batch, horizon=1):
        """
            make step on timeline with new data
        :param batch: new data
        :param horizon: prediction horizon
        :return:
        """

        for series in zip(*batch):  # очередная порция данных
            self.time += 1
            if not (self.time % 100000):
                print('Живем!!', self.stakes / self.time)

            pred = np.zeros((horizon, self.nod+1, self.max_time_scale), dtype=bool)

            # module with T=i (quantization period) starts
            for j in range(len(series)):
                for i in range(1, self.max_time_scale+1):
                    if not (self.time % i):
                        pred[0, j, i-1] = self.hists[j][i - 1].make_prediction()
                        self.hists[j][i - 1].step(series[j])
                        self.accuracy[j][i - 1].append(self.hists[j][i - 1].accuracy / self.time)
            prediction = self.voting(pred[0, :, :])
            self.stakes[~(prediction ^ series)] += 1

    def formBinarySeries(self, batch, *args, **kwargs):
        """
         (next: ma - Moving Average)
         Forms binary series of vector f,f',f'',... of
         length nod(number of derivatives)+1 like
         [ [ ma>f(t0), f'(t0)>0, f''(t0)>0, ...],
           ...................................
           [ ma>f(t), f'(t)>0, f''(t)>0, ...] ]
        :return: list of bitarray batches with for every nod+1 bin series
        """
        self.alpha = kwargs.get('alpha', self.alpha)  # coeff of smoothness
        for i in range(self.nod + 1):
            self.series[i] = bitarray()

        for (ma, f) in zip(ema(batch, alpha=self.alpha, ma=self.ft_1[0]), batch):
            self.series[0].append(ma > f)
            fi_dt = f
            for i in range(1, self.nod + 1):
                dfi_dt = fi_dt - self.ft_1[i]
                self.series[i].append(dfi_dt > 0)
                self.ft_1[i] = fi_dt
                fi_dt = dfi_dt
                # print(series)
        self.ft_1[0] = ma

        return self.series


if __name__ == '__main__':

    files = ['data/points_s.csv']#['C25600000.zip']
    stream = FileStream(files)
    forecast_module = FModule(stream, nod=2, max_time_scale=1)
    futures=[]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(1, 6):
            futures.append(executor.submit(Visual.save_generated_plot, "time_scale_{}.mp4".format(i),
                            predict, Stream.fileStream, files, nod=nod, max_time_scale=i))
            print(futures[i-1].running())
