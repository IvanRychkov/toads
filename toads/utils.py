import numpy as np
from collections.abc import Iterable


def float_equal(a, b, threshold=1e-6):
    """Проверяет равенство двух float-значений."""
    return np.abs(a - b) < threshold


def dict_agg(d, agg_func=np.mean):
    """Вызывает функцию агрегации для каждого элемента словаря."""
    return {k: agg_func(v) if issubclass(v.__class__, Iterable) and not isinstance(v, str) else v
            for k, v in d.items()}


def printif(*args, condition=True, **kws):
    """Печатает при условии condition.
    Полезно в связке с параметров verbose."""
    if condition:
        print(*args, **kws)


__all__ = ['dict_agg', 'float_equal', 'printif']
