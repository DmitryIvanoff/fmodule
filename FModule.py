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
        ma = alpha*f+(1.-alpha)*ma
        yield ma


class FModule:

    def __init__(self, *args, **kwargs):

        self.nod = kwargs.get('nod', 2)
        self.max_time_scale = kwargs.get('max_time_scale', 1)
        self.max_offset = np.empty((self.nod+1,self.max_time_scale ),dtype='uint16')
        self.bufflength = np.empty((self.nod+1,self.max_time_scale ),dtype='uint16')

        # initialize sizes of histograms
        for i in range(self.max_time_scale):
            self.max_offset[:, i] = kwargs.get('max_offset', 256)
            self.bufflength[:, i] = kwargs.get('bufflength', 8)     # bit length

        dt = np.dtype('uint32')          # numpy.dtype of hists
        self.time = np.zeros(self.nod+1, dtype=dt)                    # time for accuracy control
        self.stakes = np.zeros((self.nod + 1,2), dtype=dt)  # count of all stakes[0] and successful[1] for binary series
        self.stakes_series = (self.nod+1)*[bitarray()]  # bitarrays for stakes series
        self.lags = np.zeros((self.nod+1,self.max_time_scale), dtype='uint8')                      # lags counters
        self.hists = [[] for i in range(self.nod+1)]            # hists of various time scales
        self.coroutines = [[] for i in range(self.nod+1)]       # coroutines
        self.accuracy = np.zeros(self.nod + 1, dtype=dt)
        for j in range(self.nod + 1):
            for i in range(self.max_time_scale):
                self.hists[j].append([])
                self.coroutines[j].append(None)
                self.lags[j][i] = i
                for k in range(i + 1):
                    self.hists[j][i].append(Hist.Histogram(self.max_offset[j, i], self.bufflength[j, i]))

        self.alpha = kwargs.get('alpha', 0.9)         # coeff of smoothness
        self.series = [bitarray()]                    # list (nod+1 length) of bitarray batches
        self.ft_1 = [None]                            # previous values [ma(t-1),f(t-1),f'(t-1),.., f(nod-1)(t-1)]
        self.tasks = [None]

        # init for derivatives series
        for i in range(self.nod):
            self.ft_1.append(0.0)
            self.series.append(bitarray())
            self.tasks.append(None)

        self.predictions = np.empty((self.nod+1, self.max_time_scale), dtype='bool')

        self.prob_0_1 = np.zeros((self.nod+1, self.max_time_scale, 2), dtype='float64')

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
        # print('lol')
        binary_batch = self.formBinarySeries(batch)
        for i in range(self.nod+1):
            # print('in cycle: {}'.format(i))
            self.tasks[i] = aio.create_task(
                                            self.task(
                                                      self.time[i:i+1], self.prob_0_1[i, :, :],
                                                      self.predictions[i, :], self.lags[i, :],
                                                      self.hists[i], self.coroutines[i],
                                                      self.stakes[i,:], self.stakes_series[i],
                                                      self.accuracy[i:i+1], binary_batch[i]
                                                     )
                                            )
        done,pending = await aio.wait(self.tasks)

        all_stakes = self.stakes[:, 0]

        if np.all(all_stakes):
            st = (self.stakes[:, 1] / self.stakes[:, 0])
        else:
            st = np.zeros(self.nod+1)

        frame = np.concatenate((st, self.accuracy/self.time))
        return frame

    def vote(self, prob, predictions, *args,**kwargs):

        """
            Voting system for various histograms (various time_scales)
            System works according to principle:
            'Vote'* happens when
            1)every hist (various time_scales for one bin series) make prediction
            2)all predictions are same
            * Vote - producing common prediction for given binary series
        :param prob: probabilities (1 and 0) for every time scale(i)  -prob[i,0:1]
        :param predictions: predictions of every distinct histogram   -predictions[i]
        :return: bool array of predictions for bin series
        """
        probabilities = np.sum(prob, axis=0)
        #loop = aio.get_running_loop()
        #loop.stop()
        pred_prob = probabilities[1] > probabilities[0]

        pred_vote = -1
        pred_sum = np.sum(predictions)
        if pred_sum == predictions.shape[0]:
            pred_vote = 1
        if pred_sum == 0:
            pred_vote = 0

        return pred_prob, pred_vote

    async def step(self, prob, predictions, hist, time_scale, value):
        """

        :param prob:     probability of 1 and 0 (np.view with shape = (max_time_scale,2))
        :param predictions:  predictions of hists with various ts (np.view with shape = (max_time_scale,1)
        :param hist:   hist for
        :param time_scale: given ts
        :param value:  next value from batch
        :return:
        """
        prob[time_scale] = np.array(hist.take_probabilities())
        predictions[time_scale] = hist.prediction
        hist.step(value)

    async def task(self, time, prob, predictions, lags, hists, coroutines, stakes, stakes_series, accuracy, batch):
        """

        :param time:
        :param prob:
        :param predictions:
        :param lags:
        :param hists:
        :param coroutines:
        :param stakes:
        :param stakes_series:
        :param accuracy:
        :param batch:
        :return:
        """
        for value in batch:  # batch
            time += 1
            if not (time % 10000):
                print('time:', time, 'accuracy(voting):', stakes[1]/stakes[0],
                      'stakes count:', stakes[0], 'accuracy(prob meth):', accuracy/time)

            # sum of probabilities from various time scale hists
            prob.fill(0)

            # array of predictions
            predictions.fill(0)

            # module with T=i (quantization period) prepares coroutine for concurrent step
            for i in range(self.max_time_scale):
                lags[i] = lags[i] % (i + 1)
                coroutines[i] = self.step(prob, predictions, hists[i][lags[i]], i, value)
            # print('in task')
            await aio.gather(*coroutines)

            # using lags
            lags -= 1

            # voting for prediction
            prob_pred, vote_pred = self.vote(prob, predictions)

            # all voting
            if vote_pred > -1:
                stakes[0] += 1
                stakes[1] += not (vote_pred ^ value)
                stakes_series.append(True)
            else:
                stakes_series.append(False)
            accuracy += not (prob_pred ^ value)

    def formBinarySeries(self, batch, *args, **kwargs):
        """
         (next: ma - Moving Average)
         Forms binary series of vector f,f',f'',... of
         length nod(number of derivatives)+1 like
         [ [ ma>f(t0), f'(t0)>0, f''(t0)>0, ...],
           ...................................
           [ ma>f(t), f'(t)>0, f''(t)>0, ...] ]
        :param batch: batch of values (f(t)) (iterable)
        :return: list of bitarray batches with for every nod+1 bin series
        """
        # self.alpha = kwargs.get('alpha', self.alpha)  # coeff of smoothness
        for i in range(self.nod + 1):
            self.series[i] = bitarray()

        if not self.ft_1[0]:
            self.ft_1[0] = batch[0]

        for f in batch:
            ma = self.alpha*f+(1.-self.alpha)*self.ft_1[0]
            self.series[0].append(f > ma)
            fi_dt = f
            for i in range(1, self.nod + 1):
                dfi_dt = fi_dt - self.ft_1[i]
                self.series[i].append(dfi_dt > 0)
                self.ft_1[i] = fi_dt
                fi_dt = dfi_dt
                # print(series)
            self.ft_1[0] = ma
        '''
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
        '''
        return self.series


async def main(**kwargs):

    files = ['data/points_s.csv']  # ['C25600000.zip']
    stream = FileStream(files, 1000)
    # loop = aio.get_running_loop()
    max_ts = 5
    forecast_module = FModule(nod=2, max_time_scale=max_ts)
    print('Поехали!!')

    async def frame_get():
        batch = stream.pop(100)
        if batch:
            # print(batch)
            f = await forecast_module.predict(batch)
            return f
        else:
            return None

    with concurrent.futures.ThreadPoolExecutor() as pool:
        pool.submit(stream.load_from_files)
        frames = []
        print('Hello! thread:', current_thread())
        while True:  # stream.eof and stream.queue.empty():
            frame = await frame_get()
            if not (frame is None):
                #print(frame)
                frames.append(frame)
            else:
                print('lol')
                break
    Visual.save_plot("forecast_module_{}.mp4".format(max_ts), frames)

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

