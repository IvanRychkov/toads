import numpy as np
import seaborn as sns
import pandas as pd
from IPython.display import display
from .image import Image


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


def first_look(df: pd.DataFrame(), scatter_matrix=True) -> None:
    """Выводит наиболее популярные сведения о датафрейме."""
    df.info()
    print('-' * 50)
    print('head()')
    display(df.head(3))
    print('-' * 50)
    print('nunique()')
    display(df.nunique())
    print('-' * 50)
    print('describe()')
    display(describe(df))
    print('-' * 50)
    print('corr()')
    display(df.corr())
    if scatter_matrix:
        sns.pairplot(df)
    print()


def print_shapes(*arrays):
    """Принимает список массивов и печатает их размеры."""
    for a in arrays:
        print(a.shape)


def plot_dist_classic(df, columns, dp_kws={}, bp_kws={}):
    """Рисует гистограммы и ящики с усами для каждого столбца датафрейма из списка."""
    for col in columns:
        with Image(st=f'Распределение признака "{col}"'):
            Image.subplot('211')
            sns.distplot(df[col], **dp_kws)
            Image.subplot('212')
            sns.boxplot(df[col], color='orange', **bp_kws)


__all__ = ['describe', 'dist_stats', 'first_look', 'na_part', 'plot_dist_classic', 'print_shapes']
