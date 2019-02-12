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
import Stream
from Stream import FileStream
from functools import reduce
import concurrent.futures
import asyncio as aio
from threading import current_thread

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

    # init for derivatives series
    for i in range(nod):
        t_1.append(0.0)
        series.append(bitarray())

    for batch in stream:
        # print(batch)
        for i in range(nod + 1):
            series[i] = bitarray()
        for (ma, f) in zip(ema(batch, alpha=alpha, ma=t_1[0]), batch):
            series[0].append(f > ma)
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
    lags = np.zeros(max_time_scale, dtype='uint8')
    stakes = np.zeros(nod + 1)
    hists = [[] for i in range(max_time_scale)]  # hists of various time scales
    for i in range(max_time_scale):
        for j in range(nod + 1):
            hists[i].append([])
            for k in range(i+1):
                hists[i][j].append(Hist.Histogram(max_offset, bufflength))
        lags[i] = i
            #accuracy[j].append([])
    print('Поехали!')
    # ------------
    for batch in formBinarySeries(stream, nod=nod):
        for series in zip(*batch):  # очередная порция данных
            time += 1
            if not (time % 100000):
                print('Живем!!', stakes / time)
                # module with T=i (quantization period) starts
            prob_0_1 = np.zeros((nod + 1, 2))
            # module with T=i (quantization period) starts

            for i in range(max_time_scale):
                lags[i] = lags[i] % (i+1)
                for j in range(len(series)):
                    prob_0_1[j] += np.array(hists[i][j][lags[i]].take_probabilities())
                    hists[i][j][lags[i]].step(series[j])

            accuracy = np.array([[hists[i][j][lags[i]].accuracy / hists[i][j][lags[i]].timestamp
                                  for i in range(max_time_scale)]
                                  for j in range(nod + 1)])
            lags -= 1
            prediction = prob_0_1[:, 1] > prob_0_1[:, 0]
            stakes[~(prediction ^ series)] += 1
        points = np.concatenate((stakes / time, accuracy[0,:]))
        yield points



class FModule:

    def __init__(self, *args, **kwargs):

        self.nod = kwargs.get('nod', 2)
        self.max_time_scale = kwargs.get('max_time_scale', 1)
        self.max_offset = np.empty((self.max_time_scale, self.nod+1),dtype='uint16')
        self.bufflength = np.empty((self.max_time_scale, self.nod+1),dtype='uint16')

        # initialize sizes of histograms
        for i in range(self.max_time_scale):
            self.max_offset[i, :] = kwargs.get('max_offset', 256)
            self.bufflength[i, :] = kwargs.get('bufflength', 8)     # bit length

        dt = np.dtype('uint32')          # numpy.dtype of hists
        self.time = 0                    # time for accuracy control
        self.stakes = np.zeros(self.nod + 1)       # count of stakes for binary series
        self.stakes_series = (self.nod+1)*[bitarray()]  # bitarrays for stakes series
        self.lags = np.zeros(self.max_time_scale, dtype='uint8')                      # lags counters
        self.hists = [[] for i in range(self.max_time_scale)]            # hists of various time scales
        self.coroutines = []

        for i in range(self.max_time_scale):
            for j in range(self.nod+1):
                self.hists[i].append([])
                for k in range(i+1):
                    self.hists[i][j].append(Hist.Histogram(self.max_offset[i, j], self.bufflength[i, j]))
                self.coroutines.append(None)
            self.lags[i] = i

        self.alpha = kwargs.get('alpha', 0.9)         # coeff of smoothness
        self.series = [bitarray()]                    # list (nod+1 length) of bitarray batches
        self.ft_1 = [None]                            # previous values [ma(t-1),f(t-1),f'(t-1),.., f(nod-1)(t-1)]

        # init for derivatives series
        for i in range(self.nod):
            self.ft_1.append(0.0)
            self.series.append(bitarray())

        self.current_value = np.zeros(self.nod+1, dtype='bool')

        self.predictions = np.empty((self.max_time_scale, self.nod + 1), dtype='bool')

        #self.prob_0_1 = np.zeros((self.max_time_scale, self.nod + 1, 2))
        self.prob_0_1 = np.zeros((self.nod + 1, 2))


    async def predict(self,batch, *args, **kwargs):
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
        # hist of given time scales will vote for prediction
        time_scales = kwargs.get('time_scales', tuple(range(1, self.max_time_scale + 1)))
        if min(time_scales) < 1:
            raise ValueError("Set correct time_scales!")

        # begin pipeline
        binary_batch = self.formBinarySeries(batch)

        for series in zip(*binary_batch):  # очередная порция данных
            self.time += 1
            if not (self.time % 100000):
                print('Живем!!', self.stakes / self.time)
            self.current_value = series
            # sum of probabilities from various time scale hists
            self.prob_0_1.fill(0)
            # array of predictions of
            self.predictions.fill(0)
            # module with T=i (quantization period) starts
            for i in range(self.max_time_scale):
                self.lags[i] = self.lags[i] % (i+1)
                for j in range(self.nod+1):
                    self.coroutines[i+j*self.max_time_scale] = (self.step(i, j))

            await aio.gather(*self.coroutines)
            #await aio.wait(set(self.coroutines))
            '''
                for j in range(len(series)):

                    self.prob_0_1[j] += np.array(self.hists[i][j][self.lags[i]].take_probabilities())
                    self.predictions[i][j] = self.hists[i][j][self.lags[i]].prediction
                    self.hists[i][j][self.lags[i]].step(series[j])
            '''
            # using lags
            self.lags -= 1

            # voting for prediction
            prediction = self.prob_0_1[:, 1] > self.prob_0_1[:, 0]

            #prediction = self.vote(self.predictions, time_scales)

            self.stakes[~(prediction ^ series)] += 1

        return self.stakes / self.time

    def vote(self, predictions, *args,**kwargs):
        """
            Voting system for various histograms (various time_scales)
        :param predictions: predictions of every distinct histogram for every binary series
        :return: bool array of predictions for bin series
        """

        r = (np.sum(predictions, axis=1) / predictions.shape[-1]) > 0.5
        return r

    async def step(self,i,j):
        self.prob_0_1[j] += np.array(self.hists[i][j][self.lags[i]].take_probabilities())
        self.predictions[i][j] = self.hists[i][j][self.lags[i]].prediction
        self.hists[i][j][self.lags[i]].step(self.current_value[j])

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
            self.series[0].append(f > ma)
            fi_dt = f
            for i in range(1, self.nod + 1):
                dfi_dt = fi_dt - self.ft_1[i]
                self.series[i].append(dfi_dt > 0)
                self.ft_1[i] = fi_dt
                fi_dt = dfi_dt
                # print(series)
        self.ft_1[0] = ma

        return self.series


async def main(**kwargs):

    files = ['data/points_s.csv']  # ['C25600000.zip']
    stream = FileStream(files,100)
    loop = aio.get_running_loop()
    max_ts = 5
    forecast_module = FModule(nod=2, max_time_scale=max_ts)
    # frames_stream = forecast_module.frames_stream
    # await Visual.aio_saving("forecast_module_{}.mp4".format(1), frames_stream)
    print('Поехали!!')
    async def frame_get():
        batch = stream.pop()
        r = await forecast_module.predict(batch)
        return r
    with concurrent.futures.ThreadPoolExecutor() as pool:
        coro = pool.submit(stream.load_from_files)
        #print(coro)
        print('Hello! thread:', current_thread())
        while not stream.eof:
            await Visual.aio_saving("forecast_module_{}.mp4".format(max_ts),frame_get)

if __name__ == '__main__':


    '''
    futures = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(1, 6):
            futures.append(executor.submit(Visual.save_generated_plot, "time_scale_{}.mp4".format(i),
                            predict, Stream.fileStream, files, nod=nod, max_time_scale=i))
            print(futures[i-1].running())
    '''

    aio.run(main())

    '''
    files = ['data/points_s.csv']  # ['C25600000.zip']
    nod = 2
    max_time_scale = 9
    Visual.save_plot("time_scale_{}.mp4".format(max_time_scale),
                     predict(Stream.fileStream(files), nod=nod, max_time_scale=max_time_scale))
    '''
