"""Модуль содержит функции-помощники для Data Science"""
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.ticker import PercentFormatter
from itertools import chain
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score


def desc(df):
    """Возвращает транспонированный describe()
    """
    return df.describe().transpose()
        
        
def distr(column):
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


def hypo(test, alpha=0.05, oneside=False, show=True):
    """Сравнивает p-value с уровнем значимости. Проверяет гипотезу.
    
    alpha: уровень значимости
    
    oneside: делит p-value пополам
    
    show: печатает либо возвращает bool
    """
    if isinstance(test, tuple):
        pv = test[1]
    else:
        pv = test.pvalue if not oneside else test.pvalue / 2
    result = pv > alpha
    if show:
        print('p-value =', pv)
        print('p-value',
              '>' if result else '<',
              alpha)
    else:
        return result
    

def na_part(data, column):
    """Возвращает долю пропусков в столбце датафрейма.
    """
    print('Доля пропусков в столбце "{}" равна {:.1%}'
          .format(column, data[column].isna().sum() / len(data)))
    
    
def multiplot(dfs: list, labels: list, vals: str, groupers: dict, af='mean', plotmap=None):
    # На случай отсутствия названий таблиц
    if len(labels) < len(dfs):
        labels = ['таблица {}'.format(x + 1) for x, _ in enumerate(dfs)]
        # По умолчанию график-линия для всего
    if not plotmap:
        plotmap = 'p' * len(groupers)
        # Проходимся по каждому группировочному столбцу данных
    for grouper, name, pt in zip(groupers.keys(), groupers.values(), plotmap):
        # Рисуем таблицу для этого столбца
        figure()
        # Задаём непрозрачность
        ap = 1.0
        # Совмещаем столбцы с названиями через zip
        for df, lb in zip(dfs, labels):
            lb = lb if lb else df.name
            if pt == 'p':
                # Делаем сводную таблицу
                pvt = df.pivot_table(index=grouper, values=vals, aggfunc=af)
                plt.plot(vals, data=pvt, label=lb, alpha=ap)
                # Немного уменьшаем непрозрачность для каждого следующего графика
                ap -= 0.09
            if pt == 's':
                ap -= 0.35
                plt.scatter(grouper, vals, data=df, label=lb, alpha=ap)
            if pt == 'b':
                plt.bar(grouper, vals, data=df, label=lb, alpha=ap)
            # Отображаем рисунок
        show(name, True, 'f')
        

def median_fill_by_cat(df, target_column, cat_column):
    """Заполнение медианой по категории (таблица, столбец с пустыми значениями, столбец с категориями)"""
    # Для каждой категории, где есть пустые значения в целевом столбце:
    for cat in df[df[target_column].isna()][cat_column].unique():
        # Заполняем в целевом столбце пропуски медианой, соответствующей этой категории
        df.loc[df.loc[:, cat_column] == cat, target_column] =\
        df.loc[df.loc[:, cat_column] == cat, target_column].fillna(
            df.loc[df.loc[:, cat_column] == cat, target_column].median())


def fill_from(row, by, what, source):
    """Возвращает значение из таблицы-источника.
    row - строка
    by - по какому признаку искать
    what - столбец, который заполняем
    source - таблица-источник
    """
    try:
        return source.at[row[by], what]
    # Если индекса нет
    except KeyError:
        return row[what]


def train_val_test(df, r_state=None, strat=None):
    """Делит датасет на три выборки в отношении 3:1:1"""
    strat = df[strat] if strat else None
    train_val, test = train_test_split(df, test_size=0.2, random_state=r_state, stratify=strat)
    train, val = train_test_split(train_val, test_size=0.25, random_state=r_state, stratify=strat)
    return train, val, test


def xy_split(data, target: str, as_list=False, xs_first=False):
    """Возвращает кортеж из признаков и целевого признака.
    
    Если получает на вход список датафреймов, возвращает список кортежей.
    -----
    data - датафрейм или список датафреймов
    target - целевой признак
    as_list - Возвращает сплошной список
    xs_first - сначала все x, затем y
    """
    if isinstance(data, (list, tuple)):
        if xs_first:
            l = np.transpose(xy_split(data, target))
            if as_list:
                return tuple(chain(l[0], l[1]))
            return l
        l = []
        for element in data:
            xy = xy_split(element, target)
            # Списком или кортежами
            l.extend(xy) if as_list else l.append(xy)
        return l
    return data.drop(target, axis=1), data[target]


def proba_thresholds(model, x_val, y_val, step=0.005):
    """Оценивает качество модели-классификатора с разными порогами вероятности"""
    observations = defaultdict(list)
    for threshold in np.arange(0, 1+step, step):
        preds = (model.predict_proba(x_val)[:, 1] > threshold).astype('int')
        observations['threshold'].append(threshold)
        observations['accuracy'].append(accuracy_score(y_val, preds))
        observations['f1'].append(f1_score(y_val, preds))
        observations['recall'].append(recall_score(y_val, preds))
        observations['precision'].append(precision_score(y_val, preds))
        observations['roc_auc'].append(roc_auc_score(y_val, preds))
    return pd.DataFrame(observations)


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
    def format_axis(axis, formatter):
        """Форматирует ось форматтером из matplotlib
        """
        if axis in [0, 'x']:
            plt.gca().xaxis.set_major_formatter(formatter)
        if axis in [1, 'y']:
            plt.gca().yaxis.set_major_formatter(formatter)


def first_look(df: 'pandas.DataFrame') -> None:
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