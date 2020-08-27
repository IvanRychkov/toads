import numpy as np
import pandas as pd

def agg_fill_by_cat(df, target_column, cat_column, aggfunc='median'):
    """Заполнение результатом aggfunc по категории (таблица, столбец с пустыми значениями, столбец с категориями)"""
    # Для каждой категории, где есть пустые значения в целевом столбце:
    for cat in df[df[target_column].isna()][cat_column].unique():
        # Заполняем в целевом столбце пропуски медианой, соответствующей этой категории
        df.loc[df.loc[:, cat_column] == cat, target_column] = \
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


def ci_strip(data, ci=0.95, subset: 'list[str]' = None):
    """Принимает набор данных и возвращает значения, лежащие в доверительном интервале.

    data: {pd.DataFrame, array-like}
    subset: [str] = список столбцов, по которым идёт отсечка."""
    # Если датафрейм
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        columns = subset if subset else df.columns
        for col in columns:  # Для каждого столбца
            df = df.loc[ci_strip(df[col], ci=ci).index]  # Оставить индексы в доверительном интервале
        return df
    else:
        lower = (1 - ci) / 2
        upper = 1 - lower
        return data[(np.quantile(data, lower) <= data) & \
                    (data <= np.quantile(data, upper))]