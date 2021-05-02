import sqlite3
import pandas as pd


class SQLocal:
    """Класс для удобных локальных SQL-запросов в pandas."""
    def __init__(self, db: str):
        """Подключается к базе данных."""
        self.con = sqlite3.Connection(db)

    def __call__(self, query: str, **kws) -> pd.DataFrame:
        """Работает с SELECT."""
        try:
            return pd.read_sql_query(sql=query,
                                     con=self.con,
                                     **kws)
        except TypeError:
            return None

    def register_table(self, name, table, if_exists='replace', **kws):
        """Добавляет таблицу в базу данных. По умолчанию заменяет существующую таблицу."""
        table.to_sql(name, self.con,
                     index=isinstance(table.index, pd.DatetimeIndex),
                     if_exists=if_exists,
                     **kws)

    def list_tables(self):
        """Перечисляет все таблицы в базе."""
        return [t[0] for t in self.con.execute('select name from sqlite_master where type = "table"').fetchall()]


__all__ = ['SQLocal']
