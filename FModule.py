"""
date: 2.02.19
Что я хочу:
Дан временной ряд. Задана история временного ряда (history_size)
Хочу с какой-то точностью предсказывать следующее значение на шаг вперед
(шаг дикретизации ряда)
TO DO:
1. Из потока данных формировать (можно ассинхронно для бин рядов и масштабов времени)
   1.1) скользящее среднее;
   1.2) бинарные ряды (f(t)>ma(f(t),f'(t) > 0, f''(t)>0, ...);
   1.3) гистограммы распределения (см. Histogram) для различных масштабов времени
      с лагами
2. Система "голосования", согласно которой выбирается прогнозируемое значение
    расчет оценок точности, скользящих средних точности и т.д.
    2.1) Единогласное голосование.Понятие "ставки" (stake) как момента когда все гистограммы
    ( для различных масштабов времени) предсказывают одинаковое значение
    2.2) Вероятность ставки, вероятность угадывания ставки
    2.3) Голосование по сумме вероятностей 1 или 0 (суммируем по всем масштабам времени)
3. Визуализация
   3.1) Графики точности ставок, точности
4. Тестирование
5. Оптимизация
6. Production
"""

import numpy as np
from bitarray import bitarray
import Histogram as Hist
import Visual
from Stream import FileStream
import concurrent.futures
import asyncio as aio
from threading import current_thread
import tools
import time as Clock
import logging

def _ema(series, alpha, ma=None):
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


def ema(value, alpha, ma=None):
    """
     Calculates exponentially weighted moving average  (https://en.wikipedia.org/wiki/Moving_average)
    :param series:
    :param alpha:
    :param ma: ma(t-1)
    :return: generator(ma(t),m(t+1),..ma(t+len(series))

    """
    if not ma:
        return value
    return alpha*value+(1.-alpha)*ma


class FModule:

    def __init__(self, *args, **kwargs):

        self.nod = kwargs.get('nod', 2)
        self.max_time_scale = kwargs.get('T_max', 1)
        self.max_offset = np.empty((self.nod+1,self.max_time_scale ),dtype='uint16')
        self.bufflength = np.empty((self.nod+1,self.max_time_scale ),dtype='uint16')

        # initialize sizes of histograms
        for i in range(self.max_time_scale):
            self.max_offset[:, i] = kwargs.get('max_offset', 256)
            self.bufflength[:, i] = kwargs.get('bufflength', 8)     # bit length

        dt = np.dtype('uint32')          # numpy.dtype of hists

        self.lags = np.zeros((self.nod+1,self.max_time_scale), dtype='uint8')                      # lags counters
        self.hists = [[] for i in range(self.nod+1)]            # hists of various time scales

        self.time = np.zeros(self.nod+1, dtype=dt)                    # time for accuracy control
        self.stakes = np.zeros((self.nod + 1,2), dtype=dt)         # count of all stakes[0] and successful[1] for binary series
        self.stakes_series = (self.nod+1)*[bitarray()]              # bitarrays for stakes series

        self.accuracy = np.zeros(self.nod + 1, dtype=dt)         # accuracy according to probabilities predictions
        self.stakes_ma = np.zeros((self.nod + 1,2), dtype='float32')  # stakes accuracy moving average
        for j in range(self.nod + 1):
            for i in range(self.max_time_scale):
                self.hists[j].append([])
                self.lags[j][i] = 0
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

        logging.info(
        """
        FModule starts with: nod={};Tmax={};
        Hist max offset: {}, bufflength: {}, moving average coeff: {} 
        """.format(
             self.nod,self.max_time_scale,
             self.max_offset,self.bufflength,
             self.alpha
          )
                     )

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
        """
        if np.all(self.time):
            all_stakes = self.stakes[:, 0]

            if np.all(all_stakes):
                st = (self.stakes[:, 1] / self.stakes[:, 0])
            else:
                st = np.zeros(self.nod+1)

            frame = np.concatenate((st, self.accuracy/self.time))
        else:
            print('sdfsdf')
            frame = np.zeros(2*(self.nod+1))
        """
        # start = Clock.process_time()
        for i in range(self.nod + 1):
            # print('in cycle: {}'.format(i))

            self.tasks[i] = aio.create_task(
                self.task(
                    i,
                    self.time[i:i + 1], self.prob_0_1[i, :, :],
                    self.predictions[i, :], self.lags[i, :],
                    self.hists[i], self.stakes[i, :],
                    self.stakes_series[i],
                    self.stakes_ma[i, :],
                    self.accuracy[i:i + 1], binary_batch[i]
                )
            )
        done, pending = await aio.wait(self.tasks)
        # elapsed = Clock.process_time() - start
        # print('tasks time: {:.5f}'.format(elapsed))
        all_stakes = self.stakes[:, 0]

        if np.all(all_stakes):
            st = (self.stakes[:, 1] / self.stakes[:, 0])
        else:
            st = np.zeros(self.nod + 1)

        frame = np.concatenate((st, self.accuracy / self.time))
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
        # probabilities prediction: just summarize probabilities for all time scales for 1 and 0
        # and choose bigger
        pred_prob = probabilities[1] > probabilities[0]
        pred_vote = -1
        pred_sum = np.sum(predictions)
        if pred_sum == predictions.shape[0]:
            pred_vote = 1
        if pred_sum == 0:
            pred_vote = 0

        return pred_prob, pred_vote

    async def task(self, name, time, prob, predictions, lags, hists,
                   stakes, stakes_series, stakes_ma, accuracy, batch):
        """
             task for each binary series
        :param name
        :param time:    timestamp for given batch
        :param prob:    probabilities from various time scale hists
        :param predictions:
        :param lags:
        :param hists:
        :param stakes:
        :param stakes_series:
        :param stakes_ma:
        :param accuracy:
        :param batch:
        :return:
        """
        batch_size = batch.length()
        for value in batch:

            # logging
            if not ((time % batch_size) or time<=0):
                msg = "binary series {}:\n".format(name)
                for i in range(self.max_time_scale):
                    for j in range(i + 1):
                        msg += "\tts{} lag{}: {:.5f}%; size: {};\n".format(
                            i + 1, i - j,
                            100.0 * hists[i][j].accuracy / hists[i][j].timestamp,
                            hists[i][j].sum)

                stakes_ma[0] = ema(100.0 * (stakes[1] / stakes[0]), 0.3, stakes_ma[0])
                stakes_ma[1] = ema(100.0 * (accuracy / batch_size), 0.3, stakes_ma[1])

                print('time:', time, 'accuracy(voting): {}%'.format(stakes_ma[0]),
                      'stakes count:', stakes[0], 'accuracy(prob meth): {}%'.format(stakes_ma[1]))

                msg += "\tstakes: {:.5f}%; all: {}; successful: {};\n".format(
                    100.0 * stakes[1] / stakes[0],
                    stakes[0],
                    stakes[1])

                msg += "\tprobability voting: {}%;\n".format(100.0 * accuracy / batch_size)
                msg += "\tmoving averages: stakes {}; probability: {};\n".format(stakes_ma[0], stakes_ma[1])

                stakes[1] = 0
                stakes[0] = 0
                accuracy[:] = 0
                logging.info('[{}]:{}'.format(time, msg))

            time += 1
            # sum of probabilities from various time scale hists
            prob.fill(0)

            # array of predictions
            predictions.fill(0)

            # module with T=i (quantization period) steps forward
            for i in range(self.max_time_scale):
                lags[i] = lags[i] % (i + 1)
                prob[i] = np.array(hists[i][lags[i]].take_probabilities())
                predictions[i] = hists[i][lags[i]].prediction
                hists[i][lags[i]].step(value)

            # using lags
            #print(lags)
            lags += 1

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
        series = [bitarray() for i in range(self.nod + 1)]

        if not self.ft_1[0]:
            self.ft_1[0] = batch[0]

        for f in batch:
            ma = self.alpha*f+(1.-self.alpha)*self.ft_1[0]
            series[0].append(f > ma)
            fi_dt = f
            for i in range(1, self.nod + 1):
                dfi_dt = fi_dt - self.ft_1[i]
                series[i].append(dfi_dt > 0)
                self.ft_1[i] = fi_dt
                fi_dt = dfi_dt
                # print(series)
            self.ft_1[0] = ma
        '''
        for (ma, f) in zip(_ema(batch, alpha=self.alpha, ma=self.ft_1[0]), batch):
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
        return series


async def main(**kwargs):

    files = kwargs.get('files', ['data/points_s.csv'])  # ['C25600000.zip']
    batch_size = kwargs.get('batch_size', 1000)
    stream = FileStream(files, kwargs.get('sqs', 10*batch_size))
    # loop = aio.get_running_loop()
    max_ts = kwargs.get('T_max',1)
    forecast_module = FModule(**kwargs)
    print('Поехали!!')

    async def frame_get():
        batch = stream.pop(batch_size)
        if batch:
            # start = Clock.process_time()
            f = await forecast_module.predict(batch)
            # elapsed = Clock.process_time() - start
            # print('batch prediction time: {:.5f}'.format(elapsed))
            return f
        else:
            return None

    with concurrent.futures.ThreadPoolExecutor() as pool:
        pool.submit(stream.load_from_files)
        frames = []
        print('Hello thread:', current_thread())
        while True:  # stream.eof and stream.queue.empty():
            frame = await frame_get()
            if not (frame is None):
                #print(frame)
                frames.append(frame)
            else:
                print('lol')
                break
    Visual.save_plot("forecast_module_{}.mp4".format(max_ts), frames)


def process_main(nod,T_max):
    files = ['data/points_s.csv', 'C25600000.zip']
    logging.basicConfig(filename='FModule_{}_{}.log'.format(nod, T_max), level=logging.INFO)
    aio.run(main(files=files, nod=nod, T_max=T_max, alpha=0.5, max_offset=512))


if __name__ == '__main__':

    futures = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(1, 2):
            futures.append(executor.submit(process_main,nod=2, T_max=2*(i+1)))
            print(futures[i-1].running())


