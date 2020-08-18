"""Модуль содержит функции-помощники для Data Science"""
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split
from itertools import chain
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from IPython.display import display
import seaborn as sns


def desc(df):
    """Возвращает транспонированный describe()
    
    Добавляет строку с долей пропусков для каждого столбца.
    """
    return df.describe().append(df.agg([na_part])).transpose()
        
        
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


def hypo(test, alpha=0.05, oneside=False, verbose=True):
    """Сравнивает p-value с уровнем значимости. Проверяет гипотезу.
    
    alpha: уровень значимости
    
    oneside: делит p-value пополам
    
    verbose: печатает либо возвращает bool
    """
    if isinstance(test, tuple):
        pv = test[1]
    else:
        pv = test.pvalue if not oneside else test.pvalue / 2
    result = pv > alpha
    if verbose:
        print('p-value =', pv)
        print('p-value',
              '>' if result else '<',
              alpha)
    else:
        return result
    

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


def agg_fill_by_cat(df, target_column, cat_column, aggfunc='median'):
    """Заполнение результатом aggfunc по категории (таблица, столбец с пустыми значениями, столбец с категориями)"""
    # Для каждой категории, где есть пустые значения в целевом столбце:
    for cat in df[df[target_column].isna()][cat_column].unique():
        # Заполняем в целевом столбце пропуски медианой, соответствующей этой категории
        df.loc[df.loc[:, cat_column] == cat, target_column] =\
        df.loc[df.loc[:, cat_column] == cat, target_column].fillna(
            df.loc[df.loc[:, cat_column] == cat, target_column].agg(aggfunc))


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
    def format_axis(axis, formatter=PercentFormatter(xmax=1, decimals=0)):
        """Форматирует ось форматтером из matplotlib"""
        if axis in [0, 'x']:
            plt.gca().xaxis.set_major_formatter(formatter)
        if axis in [1, 'y']:
            plt.gca().yaxis.set_major_formatter(formatter)


def first_look(df: 'pd.DataFrame') -> None:
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


def ci_strip(data, ci=0.95, subset: 'list(str)'=None):
    """Принимает набор данных и возвращает значения, лежащие в доверительном интервале.
    
    data: {pd.DataFrame, array-like}
    subset: [str] = список столбцов, по которым идёт отсечка."""
    # Если датафрейм
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        columns = subset if subset else df.columns
        for col in columns:                              # Для каждого столбца
            df = df.loc[ci_strip(df[col], ci=ci).index]  # Оставить индексы в доверительном интервале
        return df
    else:
        lower = (1 - ci) / 2
        upper = 1 - lower
        return data[(np.quantile(data, lower) <= data) &\
                    (data <= np.quantile(data, upper))]


def smape(true, preds):
    """Symmetric Mean Absolute Percentage Error, симметричное среднее абсолютное процентное отклонение.
    
    Она похожа на MAE, но выражается не в абсолютных величинах, а в относительных. Одинаково учитывает масштаб и целевого признака, и предсказания."""
    return np.nanmean(np.abs(true - preds) / ((np.abs(true) + np.abs(preds)) / 2))


def train_model_cv(model, x, y, scorer, features=None, cv=5, **cv_kws):
    """Обучает модель с возможностью выбора признаков. Проверяет кросс-валидацией. Возвращает среднее значение метрики качества."""
    if features is None:
        features = x.columns
    return np.abs(np.mean(cross_val_score(model, x[features], y, cv=cv, scoring=scorer, **cv_kws)))


def batch_train_cv(models, train_func: train_model_cv, names=None, greater_is_better=False, **train_kws):
    """Обучает модели из списка, собирает результаты в Series."""
    scores = []
    for model in models:
        scores.append(train_func(model, **train_kws))
    return pd.Series(scores, index=models if names is None else names, name='score').sort_values(ascending=not greater_is_better)


def dist_compare(true, preds, hypothesis=True, image='dist', **img_kws):
    """Рисует распределения ответов модели и реальных значений целевого признака.
    Проверяет сходство выборок критерием Манна-Уитни."""
    if image:
        with Image(legend='a' if image=='dist' else None, **img_kws):
            if image == 'dist':
                sns.distplot(true, label='true')
                sns.distplot(preds, label='preds')
            if image=='box':
                sns.boxplot(data=[true, preds], orient='h')
                plt.yticks([0, 1], ['true', 'preds'])
    if hypothesis:
        hypo(mannwhitneyu(true, preds))