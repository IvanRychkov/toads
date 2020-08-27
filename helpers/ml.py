from sklearn.model_selection import train_test_split, cross_val_score
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from itertools import chain
import numpy as np
import pandas as pd


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
    for threshold in np.arange(0, 1 + step, step):
        preds = (model.predict_proba(x_val)[:, 1] > threshold).astype('int')
        observations['threshold'].append(threshold)
        observations['accuracy'].append(accuracy_score(y_val, preds))
        observations['f1'].append(f1_score(y_val, preds))
        observations['recall'].append(recall_score(y_val, preds))
        observations['precision'].append(precision_score(y_val, preds))
        observations['roc_auc'].append(roc_auc_score(y_val, preds))
    return pd.DataFrame(observations)


def smape(true, preds):
    """Symmetric Mean Absolute Percentage Error, симметричное среднее абсолютное процентное отклонение.

    Она похожа на MAE, но выражается не в абсолютных величинах, а в относительных. Одинаково учитывает масштаб и целевого признака, и предсказания."""
    return np.nanmean(np.abs(true - preds) / ((np.abs(true) + np.abs(preds)) / 2))


def train_model_cv(model, x, y, scorer, features=None, cv=5, **cv_kws):
    """Обучает модель с возможностью выбора признаков. Проверяет кросс-валидацией. Возвращает среднее значение
    метрики качества. """
    if features is None:
        features = x.columns
    return np.abs(np.mean(cross_val_score(model, x[features], y, cv=cv, scoring=scorer, **cv_kws)))


def batch_train_cv(models, train_func: train_model_cv, names=None, greater_is_better=False, **train_kws):
    """Обучает модели из списка, собирает результаты в Series."""
    scores = []
    for model in models:
        scores.append(train_func(model, **train_kws))
    return pd.Series(scores, index=models if names is None else names, name='score').sort_values(ascending=not greater_is_better)


# TODO: __all__
