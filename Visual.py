import matplotlib as mpl
#mpl.use('GTK3Agg')
#mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import numpy as np
from matplotlib.animation import FFMpegWriter
import asyncio as aio

class Painter:

    def __init__(self, axes):
        self.ax = axes

    def __call__(self, frame, *args):
        pass


class HistPainter:

    def drawHist(self, hist, ax, title, save=False, *args, **kwargs):
        '''
        :param hist:
        :return:
        '''
        cmap = kwargs.get('cmap', plt.cm.afmhot)
        norm = kwargs.get('norm', mpl.colors.LogNorm(vmin=0.0000001, vmax=np.sum(hist)))
        # hist =(hist)/np.sum(hist))
        # norm = cm.colors.Normalize(vmax=np.sum(hist), vmin=0)
        # print(hist)
        # print(np.amax(hist))
        im = ax.imshow(hist,  # interpolation='bilinear',
                        cmap=cmap,
                        origin='lower',
                        extent=[0, hist.shape[0], 0, hist.shape[1]],
                        norm=norm,
                        aspect='auto')
        ax.set(xlabel='time', ylabel='value of buffer', title=title)
        if save:
            plt.imsave('{}.png'.format(title), hist, dpi=100, cmap=cmap,
                       origin='lower', format='png')
        return im

    def draw_hists(self, hists):
        fig, axes = plt.subplots(len(hists), 1)
        fig.set_size_inches(4*len(hists), 20)
        for i in range(len(hists)):
            image = self.drawHist(hists[i], axes[i], "Histogram of Binary f'{}".format(i), save=True)
            plt.colorbar(image, ax=axes[i])
        fig.savefig('Hists.png', dpi=100, format='png')


class PlotPainter(Painter):

    def __init__(self, axes, data, ylim, xlim, fmt='',time=0):
        super().__init__(axes)
        self.time = time
        self.data = [[v0] for v0 in data]
        self.lines = []
        self.size = xlim[-1]
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.x = np.array([0])
        for i in range(len(data)):
            self.lines += self.ax.plot(self.data[i], fmt, label=str(i))
        self.ax.grid(True)
        self.ax.legend()

    def __call__(self, data, *args):
        """

        :param data:
        :param args:
        :return:
        """
        self.time += 1
        if self.time > self.size/2:
            self.ax.set_xlim(self.time-self.size/2, self.time+self.size/2)
        for i in range(len(self.lines)):
            self.data[i].append(data[i])
            self.x = np.arange(0, len(self.data[i]))
            self.lines[i].set_data(np.array([self.x, self.data[i]]))
        return self.lines


def save_plot(filename, plot, *args, **kwargs):
    """
        saves drawing of plot into movie
    :param filename: string
    :param plot:   iterable
    :param args:
    :param kwargs: writer; painter (PlotPainter type);dpi;fps;bitrate;ylim;xlim
    :return: None
    """
    fig, ax = plt.subplots()
    dpi = kwargs.get('dpi',130)
    fps = kwargs.get('fps',100)
    bitrate = kwargs.get('bitrate', 1800)
    ylim = kwargs.get('ylim', (0, 1))
    xlim = kwargs.get('xlim', (0, 35000))
    writer = kwargs.get('writer', animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=bitrate))
    painter = kwargs.get('painter', PlotPainter(ax, data=plot[0], ylim=ylim, xlim=xlim))
    with writer.saving(fig, filename, dpi):
        for frame in plot:
            painter(frame)
            writer.grab_frame()


async def aio_saving(filename, callback, **kwargs):
    """
        'asynchronous' frames saving (i.e. using async/await)
    :param filename:
    :param callback:  coroutine for getting frames from outside returns iterable
    :param kwargs:  writer; painter (PlotPainter type);dpi;fps;bitrate;ylim;xlim
    :return: None
    """
    fig, ax = plt.subplots()
    dpi = kwargs.get('dpi', 50)
    fps = kwargs.get('fps', 100)
    bitrate = kwargs.get('bitrate', 1800)
    ylim = kwargs.get('ylim', (0, 1))
    xlim = kwargs.get('xlim', (0, 35000))
    frame = await callback()
    writer = kwargs.get('writer', animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=bitrate))
    painter = kwargs.get('painter', PlotPainter(ax, data=frame, ylim=ylim, xlim=xlim))
    with writer.saving(fig, filename, dpi):
        while True:
            frame = await callback()
            painter(frame)
            writer.grab_frame()
