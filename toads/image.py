import functools
import os.path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


class Img:
    """Даёт доступ к функциям для удобной отрисовки, поддерживает конструкцию with...as.
    """

    def __init__(self, st=None, x=15, y=4,
                 grid=False, tight=False, legend=False,
                 dpi=200, no_show=False):
        self.x = x
        self.y = y
        self.st = st
        self.grid = grid
        self.tight = tight
        self.legend = legend
        self.dpi = dpi
        self.no_show = no_show

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.show()

    @staticmethod
    def figure(x, y, dpi):
        """Инициализирует рисунок в приятном для глаза разрешении
        и оптимальном размере, который можно задать при необходимости
        """
        plt.gcf().set_size_inches(x, y)
        plt.gcf().set_dpi(dpi)

    @functools.wraps(plt.savefig)
    def savefig(self, fname, **kws):
        """Best called before show()."""
        if 'dpi' not in kws:
            kws['dpi'] = 'figure'
        plt.savefig(fname, **kws)

    def show(self):
        """Поможет в одну строчку воспользоваться частыми функциями pyplot
        """
        self.figure(self.x, self.y, self.dpi)

        if len(plt.gcf().axes) != 0:
            if self.tight:
                plt.tight_layout(pad=2.5)
            if self.st:
                plt.suptitle(self.st)
            if self.grid:
                plt.grid()
            if self.legend == 'a':
                plt.legend()
            if self.legend == 'f':
                plt.figlegend()
            if not self.no_show:
                plt.show()
        plt.close('all')

    @staticmethod
    def subplot(rows, cols, pos, title=None, sx=True):
        """Функция, которая добавляет на рисунок координатную плоскость с sharex по умолчанию
        """
        if sx:
            plt.subplot(rows, cols, pos, sharex=plt.gca(), title=title)
        else:
            plt.subplot(rows, cols, pos, title=title)

    @staticmethod
    def labels(x=None, y=None, x_kws=None, y_kws=None):
        """Даёт названия осям графика, kws обращаются к kwargs названий
        --------
        0 убирает названия
        """
        if x:
            plt.xlabel(x, **x_kws if x_kws else {}) if x != 0 else plt.xlabel(None)
        if y:
            plt.ylabel(y, **y_kws if y_kws else {}) if y != 0 else plt.ylabel(None)

    @staticmethod
    def format_axis(axis, formatter=PercentFormatter(xmax=1, decimals=0)):
        """Форматирует ось форматтером из matplotlib"""
        if axis in [0, 'x']:
            plt.gca().xaxis.set_major_formatter(formatter)
        if axis in [1, 'y']:
            plt.gca().yaxis.set_major_formatter(formatter)


__all__ = ['Img']
