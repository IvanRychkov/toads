from sqlalchemy import Table
from sqlalchemy.engine import Engine
from typing import List


def insert_into_table(table: Table, db: Engine):
    """Создаёт функцию, которая принимает батч данных и вставляет его в таблицу."""

    def insert_func(data_batch: List[dict]):
        with db.begin():
            table.insert().values(data_batch).execute()

    return insert_func


__all__ = ['insert_into_table']
