import sqlite3
import pandas as pd


class SQLocal:
    """Класс для удобных локальных SQL-запросов в pandas."""
    def __init__(self, db: str):
        """Подключается к базе данных."""
        self.con = sqlite3.Connection(db)

    def __call__(self, query: str, **kws) -> pd.DataFrame:
        """Работает с SELECT."""
        return pd.read_sql_query(sql=query,
                                 con=self.con,
                                 **kws)

    def register_table(self, name, table, if_exists='replace', **kws):
        """Добавляет таблицу в базу данных. По умолчанию заменяет существующую таблицу."""
        table.to_sql(name, self.con,
                     index=isinstance(table.index, pd.DatetimeIndex),
                     if_exists=if_exists,
                     **kws)


__all__ = ['SQLContext']
