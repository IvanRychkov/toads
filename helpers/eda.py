import numpy as np
import seaborn as sns
import pandas as pd
from IPython.display import display


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


def desc(df):
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


def first_look(df: pd.DataFrame()) -> None:
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
    display(df.describe().transpose())
    print('-' * 50)
    print('corr()')
    display(df.corr())
    sns.pairplot(df)
    print()


def print_shapes(*arrays):
    """Принимает список массивов и печатает их размеры."""
    for a in arrays:
        print(a.shape)


__all__ = ['desc', 'dist_stats', 'first_look', 'na_part', 'print_shapes']
