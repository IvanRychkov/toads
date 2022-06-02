from sqlalchemy import Table
from sqlalchemy.engine import Engine
from typing import List
from sqlalchemy.dialects.postgresql import insert


def insert_into_table(table: Table, db: Engine):
    """Создаёт функцию, которая принимает батч данных и вставляет его в таблицу."""

    def insert_func(data_batch: List[dict]):
        with db.begin():
            table.insert().values(data_batch).execute()

    return insert_func


def upsert_into_postgres_table(table: Table, db: Engine, index_elements: List[str],
                               update_values_on_conflict: List[str]):
    """Try to insert rows into Postgres table"""

    def upsert_func(data_batch: List[dict]):
        with db.begin():
            insert_stmt = insert(table).values(data_batch)
            stmt = insert_stmt.on_conflict_do_update(
                index_elements=index_elements,
                set_={field: insert_stmt.excluded[field] for field in update_values_on_conflict},
            )
            stmt.execute()

    return upsert_func


__all__ = ['insert_into_table', 'upsert_into_postgres_table']
