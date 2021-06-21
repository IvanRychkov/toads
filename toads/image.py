import os.path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


class Img:
    """Даёт доступ к функциям для удобной отрисовки, поддерживает конструкцию with...as.
    """

    def __init__(self, st=None, x=15, y=4,
                 grid=False, tight=False, legend=False,
                 dpi=200, save_only=False, name='img.png', **save_kws):
        self.x = x
        self.y = y
        self.st = st
        self.grid = grid
        self.tight = tight
        self.legend = legend
        self.save_kws = save_kws
        self.dpi = dpi
        self.save_only = save_only
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if not self.save_only:
            self.show()

    @staticmethod
    def figure(x, y, dpi=200):
        """Инициализирует рисунок в приятном для глаза разрешении
        и оптимальном размере, который можно задать при необходимости
        """
        plt.gcf().set_size_inches(x, y)
        plt.gcf().set_dpi(dpi)

    def show(self, silent=False):
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
            # Сохраняем картинку
            if self.save_kws:
                # Если нет имени, то в корень
                if not self.save_kws['fname']:
                    target_folder = 'tmp/plots'
                    os.makedirs(target_folder, exist_ok=True)
                    self.save_kws['fname'] = os.path.join(target_folder, self.name)
                plt.savefig(**self.save_kws)
            if not silent:
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
