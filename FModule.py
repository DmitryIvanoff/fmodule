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
        # print(batch)
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
        # print(series)
        t_1[0] = ma
        yield series


def predict(stream, *args, **kwargs):
    """
        Gets the latest value(batch) from stream
        passes it to Binarizer( formBinarySeries
    :param stream:  iterable
    :param args:
    :param kwargs:
    :return: prediction generator
    """
    # initialization of buffers and histograms
    nod = kwargs.get('nod', 2)
    max_offset = kwargs.get('max_offset', 256)
    bufflength = kwargs.get('bufflength', 8)  # bit length
    dt = np.dtype('uint32')
    max_time_scale = kwargs.get('nod', 1)
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
    '''
    painter1 = Visual.PlotPainter(ax1, data=[[0] for i in range(nod + 1)], ylim=(0, 1), xlim=(0, 100000))
    writer = Visual.animation.FFMpegWriter(fps=35, metadata=dict(artist='Me'), bitrate=1800)
    anim1 = Visual.animation.FuncAnimation(fig, painter1,
                                          frames=predict(fileStream(files), nod=nod, max_time_scale=1),
                                          interval=20,save_count=10)
    Visual.plt.show()
    '''
    futures=[]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(1, 6):
            futures.append(executor.submit(Visual.save_generated_plot, "time_scale_{}.mp4".format(2*i),
                              predict,Stream.fileStream,files, nod=nod, max_time_scale=2*i))
            print(futures[i-1].running())

  #  Visual.save_generated_plot("time_scale_{}.mp4".format(2),
  #                             predict,Stream.fileStream,files,nod=nod,max_time_scale=2)
