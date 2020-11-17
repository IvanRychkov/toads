import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


class Img:
    """Даёт доступ к функциям для удобной отрисовки, поддерживает конструкцию with...as.
    -----
    **showparams: st=None, grid=False, legend=None, tight=False
    """

    def __init__(self, st=None, x=15, y=4,
                 grid=False, tight=False, legend=False,
                 ts_data_col=None, ts_n_ticks=20, ts_autofmt_x=True):
        self.x = x
        self.y = y
        self.st = st
        self.grid = grid
        self.tight = tight
        self.legend = legend
        self.ts_data = ts_data_col
        self.ts_n_ticks = ts_n_ticks
        self.ts_autofmt_x = ts_autofmt_x

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        self.show()

    @staticmethod
    def figure(x, y):
        """Инициализирует рисунок в приятном для глаза разрешении
        и оптимальном размере, который можно задать при необходимости
        """
        plt.gcf().set_size_inches(x, y)
        plt.gcf().set_dpi(200)

    def show(self):
        """Поможет в одну строчку воспользоваться частыми функциями pyplot
        """
        self.figure(self.x, self.y)

        if len(plt.gcf().axes) != 0:
            if (self.ts_data is not None) and self.ts_n_ticks:
                Img.time_series_format(date_col=self.ts_data,
                                       n_ticks=self.ts_n_ticks,
                                       autofmt_x=self.ts_autofmt_x)
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

    @staticmethod
    def time_series_format(date_col, n_ticks, execute=True, autofmt_x=True):
        """Formats x-axis to fit time series data without breaking it. Uses index as datetime column."""
        # Получаем тики на равном расстоянии
        ticks = np.linspace(0, date_col.shape[0] - 1, n_ticks).round().astype(int)
        # Берём названия дат из индекса
        labels = date_col[ticks] if isinstance(date_col, pd.DatetimeIndex)\
            else date_col.index[ticks]
        # По умолчанию функция сразу форматирует тики
        if execute:
            plt.xticks(ticks, labels)
            if autofmt_x:
                plt.gcf().autofmt_xdate()
        return ticks, labels


__all__ = ['Img']
