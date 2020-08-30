class AQuery:
    """Assisted query composer for SQL.
    Each query is an object allowing step-by-step construction of SQL queries."""
    __ORDER = {True: 'ASC', False: 'DESC'}
    __SELECT_DISTINCT = {True: 'SELECT DISTINCT', False: 'SELECT'}

    def __init__(self, lowercase=False):
        self.query = str()
        self._is_empty = True
        self.lowercase = lowercase

    def _append(self, clause: str = '', value: str = '', postfix='', semicolon=False):
        """Add text to query."""
        if self._is_empty:
            self._is_empty = False
        else:
            self.query += ';\n' if semicolon else '\n'
        # resulting line
        self.query += ' '.join([clause, str(value), postfix]).strip()
        return self

    def select(self, target: 'list[str]' = '*', distinct=False):
        """SELECT target columns."""
        target = ', '.join(target)
        clause = self.__SELECT_DISTINCT[distinct]
        return self._append(clause, target)

    def select_as(self, target_map: 'dict[str, str]', distinct=False):
        """SELECT dict keys AS dict values."""
        # Select clause
        clause = self.__SELECT_DISTINCT[distinct]
        # Add clause with AS mapping for each column, ignoring those with None value.
        return self._append(clause, ', '.join(
            [f'{col} AS "{alias if alias else col}"' for col, alias in target_map.items()]))

    def from_table(self, table: str):
        """FROM table."""
        return self._append('FROM', table)

    def where(self, condition: str):
        """WHERE condition.
        Can be used with LIKE (AQuery.like()).
        Better write conditions manually to avoid mistakes.
        """
        return self._append('WHERE', condition)

    @staticmethod
    def like(col: str, value: str):
        """LIKE clause for finding strings.
        Used with % and _ wildcards."""
        return f"{col} LIKE '{value}'"

    @staticmethod
    def is_null(col: str) -> str:
        return f'{col} IS NULL'

    @staticmethod
    def not_null(col: str) -> str:
        return f'{col} IS NOT NULL'

    @staticmethod
    def between(col, start, end, dtype=float) -> str:
        """Implements BETWEEN range condition for column.
        ---
        BETWEEN is case-sensitive."""
        # Todo: types.
        try:
            start, end = [dtype(x) for x in [start, end]]
        except ValueError:
            start, end = [f"'{char}'" for char in (start, end)]
        return f'{col} BETWEEN {start} AND {end}'

    # TODO
    # def create_table(self, name: str):
    #    return self._append('CREATE TABLE', name)

    def also(self):
        """Chain multiple queries together."""
        return self._append(semicolon=True)

    def order_by(self, col, ascending=True):
        return self._append('ORDER BY', col, self.__ORDER[ascending])

    def limit(self, n):
        return self._append('LIMIT', n)

    # @staticmethod
    # def case(conditions: 'dict[when, then]', as_what=None):
    #     return f"""CASE
    #     {conditions}

    def __str__(self):
        return self.query.lower() if self.lowercase else self.query +\
                                                         'Empty AQuery' if self._is_empty else ';'

    def __repr__(self):
        return self.__str__()


__all__ = ['AQuery']
