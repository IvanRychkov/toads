import numpy as np
from collections.abc import Iterable
from functools import wraps

def between(x, interval=(0, 1)):
    """Возвращает True, если x находится в интервале."""
    return interval[0] <= x <= interval[1]


def conditional(condition=True):
    """Calls function in condition is True."""
    def upper(func):
        @wraps(func)
        def wrapper(*args, **kws):
            if condition:
                return func(*args, **kws)
            pass
        return wrapper
    return upper


def float_equal(a, b, threshold=1e-6):
    """Проверяет равенство двух float-значений."""
    return np.abs(a - b) < threshold


def dict_agg(d, agg_func=np.mean):
    """Вызывает функцию агрегации для каждого элемента словаря."""
    return {k: agg_func(v) if issubclass(v.__class__, Iterable) and not isinstance(v, str) else v
            for k, v in d.items()}


def overlaps_with(subset, superset):
    """Проверяет, принадлежит ли хотя бы один элемент первого множества второму."""
    return any([element in superset
                for element in subset])


def printif(*args, condition=True, **kws):
    """Печатает при условии condition.
    Полезно в связке с параметром verbose."""
    if condition:
        print(*args, **kws)


def snake_case(s):
    """Преобразование строк из camelCase в snake_case."""
    return ''.join('_' + c.lower()
                   if all([i != 0,
                           c.isupper(),
                           s[i - 1].islower()])
                   else c.lower()
                   for i, c in enumerate(s))


__all__ = ['between', 'dict_agg', 'float_equal', 'overlaps_with', 'printif', 'snake_case']
