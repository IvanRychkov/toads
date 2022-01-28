import numpy as np
import seaborn as sns
import pandas as pd
from .image import Img
import matplotlib.pyplot as plt


def na_part(data, verbose=False):
    """Агрегирует долю пропусков в объектах pandas.

    verbose - печатает или возвращает значение
    """
    part = data.isna().sum() / len(data)
    if verbose:
        print('Доля пропусков в столбце "{}" равна {:.1%}'
              .format(data.name, part))
    else:
        return part


def describe(df):
    """Возвращает транспонированный describe()

    Добавляет строку с долей пропусков для каждого столбца.
    """
    return df.describe().append(df.agg([na_part])).transpose()


def dist_stats(column):
    """Возвращает словарь с характеристиками распределения:
    - Среднее
    - Медиана
    - Дисперсия
    - Стандартное отклонение
    """
    return {
        'mean': np.mean(column),
        'median': np.median(column),
        'var': np.var(column),
        'std': np.sqrt(np.var(column)),
    }

def print_shapes(*arrays):
    """Принимает список массивов и печатает их размеры."""
    for a in arrays:
        print(a.shape)


def plot_dist_classic(df, columns, dp_kws={}, bp_kws={}):
    """Рисует гистограммы и ящики с усами для каждого столбца датафрейма из списка."""
    for col in columns:
        with Img(f'Распределение признака "{col}"'):
            Img.subplot(2, 1, 1)
            sns.distplot(df[col], **dp_kws)
            Img.subplot(2, 1, 2)
            sns.boxplot(df[col], color='orange', **bp_kws)


def plot_time_series(data, n_ticks=15, plot_func=sns.lineplot, format_axis=True, **plot_kws):
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError('data.index must be instance of "pandas.DateTimeIndex".')
    if format_axis:
        # Если мало тиков, то тиков мало
        if n_ticks > data.shape[0]:
            n_ticks = data.shape[0]
        ticks = np.linspace(0, data.shape[0] - 1, n_ticks).round().astype(int)
        # Берём названия дат из индекса
        labels = data.index[ticks]
        plt.xticks(ticks, labels)
        plt.gcf().autofmt_xdate()

    plot_data = data.reset_index(drop=True)
    plot_data.index.rename('datetime', inplace=True)

    if isinstance(plot_data, pd.Series):
        return plot_func(x=plot_data.index,
                         y=plot_data,
                         legend=False,
                         **plot_kws)
    return plot_func(data=plot_data,
                     legend='auto',
                     **plot_kws)


__all__ = ['describe', 'dist_stats', 'first_look', 'na_part',
           'plot_dist_classic', 'plot_time_series', 'print_shapes']
