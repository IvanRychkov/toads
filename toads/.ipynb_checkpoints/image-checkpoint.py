import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


class Image:
    """Даёт доступ к функциям для удобной отрисовки, поддерживает конструкцию with...as.
    -----
    **showparams: st=None, grid=False, legend=None, tight=False
    """

    def __init__(self, x=15, y=4, **showparams):
        self.x = x
        self.y = y
        self.showparams = showparams

    def __enter__(self):
        self.figure(self.x, self.y)
        return self

    def __exit__(self, type, value, traceback):
        self.show(**self.showparams)

    def figure(self, x=15, y=4):
        """Инициализирует рисунок в приятном для глаза разрешении
        и оптимальном размере, который можно задать при необходимости
        """
        plt.figure(figsize=(x, y), dpi=200)

    def show(self, st=None, grid=False, legend=None, tight=False):
        """Поможет в одну строчку воспользоваться частыми функциями pyplot
        """
        if len(plt.gcf().axes) != 0:
            if tight: plt.tight_layout(pad=2.5)
            if st: plt.suptitle(st)
            if grid: plt.grid()
            if legend == 'a': plt.legend()
            if legend == 'f': plt.figlegend()
            plt.show()
        plt.close()

    @staticmethod
    def subplot(pos, title=None, sx=True):
        """Функция, которая добавляет на рисунок координатную плоскость с sharex по умолчанию
        """
        if sx:
            plt.subplot(pos, sharex=plt.gca(), title=title)
        else:
            plt.subplot(pos, title=title)

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


__all__ = ['Image']
