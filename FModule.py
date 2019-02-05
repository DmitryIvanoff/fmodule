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
import fileinput as fi
from bitarray import bitarray
from zipfile import ZipFile
import random as rand
import Histogram as Hist
import Visual
from functools import reduce


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


def open_hook(filename, mode):
    '''

    :param filename:
    :param mode:
    :return:
    '''
    if filename.endswith(('.csv', '.txt')):
        return open(filename, mode)
    if filename.endswith('.zip'):
        try:
            zf = ZipFile(filename)
            return zf.open(filename.replace('.zip', '.csv'),mode=mode)
        except Exception as e:
            raise e


def fileStream(files, batch_size=32, *args, **kwargs):
    """
     Generates batches from file lines (expected float number per line)
    :param files:
    :param batch_size:
    :return: numpy.ndarray
    """
    eof = False
    with fi.input(files=files, mode='r', openhook=open_hook) as f:
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
            yield np.array(r)


def formBinarySeries(stream, *args, **kwargs):
    """
     (next: ma - Moving Average)
     Forms binary series of vector f,f',f'',... of
     length nod(number of derivatives)+1 like
     [ [ ma>f(t0), f'(t0)>0, f''(t0)>0, ...],
       ...................................
       [ ma>f(t), f'(t)>0, f''(t)>0, ...] ]
    :param stream: Stream of data (stream of numbers batches as for beginning)
    :return: generator of batches
    """
    alpha = kwargs.get('alpha', 0.9)   # coeff of smoothness
    nod = kwargs.get('nod', 2)  # число производных (f'(t),f''(t) as default)

    # init for ma series
    series = [bitarray()]                         # series = [(ma>f(t),f'(t)>0,f''(t)>0,...] , len(series)=nod+1
    t_1 = [None]                          # previous values [ma(t-1),f(t-1),v(t-1)]
    ma = None

    # init for derivatives series
    for i in range(nod):
        t_1.append(0.0)
        series.append(bitarray())
    for batch in stream:
        #print(batch)
        for i in range(nod+1):
            series[i] = bitarray()
        for (ma, f) in zip(ema(batch, alpha=alpha, ma=t_1[0]), batch):
            series[0].append(ma > f)
            fi_dt = f
            for i in range(1, nod+1):
                dfi_dt = fi_dt - t_1[i]
                series[i].append(dfi_dt > 0)
                t_1[i] = fi_dt
                fi_dt = dfi_dt
        #print(series)
        t_1[0] = ma
        yield series


def acc_gen():
    pass


def predict( stream, *args, **kwargs):
    """

    :param args:
    :param kwargs:
    :return:
    """
    # initialization of buffers and histograms
    nod = kwargs.get('nod', 2)
    draw = kwargs.get('draw', False)
    max_offset = kwargs.get('max_offset', 256)
    bufflength = kwargs.get('bufflength', 8)  # bit length
    dt = np.dtype('uint32')
    max_time_scale = 10
    time = 0
    hists = []
    accuracy = []
    stakes = np.zeros(3)
    prediction = [False for i in range(nod+1)]
    for i in range(max_time_scale):
        hists.append(Hist.Histogram(nod+1, max_offset, bufflength))
        accuracy.append([])
    print('Поехали!')
    # ------------
    for batch in formBinarySeries(stream, nod=nod):
        for series in zip(*batch):  # очередная порция данных
            time += 1
            #print(series)
            if not (time % 100000):
                print('Живем!!', stakes/time)
            # module with T=i (quantization period) starts
            pred = []
            for i in range(1, max_time_scale + 1):
                if not (time % i):
                    pred.append(np.array(hists[i - 1].step(series)))
                    accuracy[i-1].append(hists[i-1].accuracy/time)
            prediction = (reduce(lambda x, y: x + y, pred) / len(pred)) > 0.5
            stakes[~prediction^series] += 1
        yield stakes/time


if __name__ == '__main__':

    files = ['data/points_s.csv']#['C25600000.zip']
    nod = 2
    fig, ax = Visual.plt.subplots(1, 1)
    #painter = Visual.PlotPainter(ax, data=[[False] for i in range(nod+1)], ylim=(0, 1.5), xlim=(0, 200))
    painter = Visual.PlotPainter(ax, data=[[0] for i in range(nod + 1)], ylim=(0, 1), xlim=(0, 200))
    anim = Visual.animation.FuncAnimation(fig, painter,
                                          frames=predict(fileStream(files), nod=nod),
                                          repeat=False,
                                          interval=5)
    Visual.plt.show()

