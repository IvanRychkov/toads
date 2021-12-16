from tqdm import tqdm
from functools import wraps


class ETL(object):
    """Class for building and executing ETL pipelines."""

    def __init__(self, extract=None, transform=None, load=None, batch_size=10000,
                 pre_execute=None, post_execute=None, count_func=None):
        """Accepts functions for each element of ETL.
        Args:
            extract (function or iterable): Sequence or iterator of atomic objects.
            transform (function) -> object: Transforms extracted atomic objects into new form.
            load (function) -> None: Loads batch of transformed objects into target database.
            pre_execute (function) -> None: Called before execution of pipeline.
            post_execute (function): Called after execution of pipeline. Serves as return value for execute() method.
            count_func (function) -> int: Function to count objects in ETL.

        pre_execute -> extract -> transform -> load -> post_execute
        """
        self._extract = None
        self.callable_extract = None
        self.extract(extract)
        self._transform = transform if transform else lambda x: x
        self._load = load if load else lambda x: None
        self.batch_size = batch_size
        self._pre_execute = pre_execute
        self._post_execute = post_execute
        self._count_func = count_func

    def __call__(self):
        """Executes ETL."""
        self.execute()

    def execute(self, extract=None):
        """Executes ETL."""
        def batch_iter(iterable):
            """Yields data in batches of arbitrary size."""
            buffer = []
            for i, v in enumerate(iterable, start=1):
                buffer.append(v)

                # If batch size reached
                if i % self.batch_size == 0 and i != 0:
                    yield buffer
                    buffer = []

            # If anything left in buffer
            if buffer:
                yield buffer

        if extract:
            self.extract(extract)

        extracted = self._extract() if self.callable_extract and not isinstance(self._extract, type) else self._extract
        try:
            iter(extracted)
        except:
            raise ValueError(f'"extract" element must {"return" if self.callable_extract else "be"} iterable.')

        # Перед запуском
        if self._pre_execute:
            self._pre_execute()

        # Считаем объекты в extract
        if self._count_func:
            extracted = tqdm(extracted, total=self._count_func())

        for batch in batch_iter(extracted):
            self._load([*map(self._transform, batch)])

        if self._post_execute:
            return self._post_execute()

    def extract(self, func):
        """Decorator for extractor function. Assigns argument 'func' as 'extract' step in pipeline."""
        self._extract = func
        self.callable_extract = callable(self._extract)
        return func

    def transform(self, func):
        """Decorator for transform function."""
        self._transform = func
        return func

    def load(self, func):
        """Decorator for load function."""
        self._load = func
        return func

    def pre_execute(self, func):
        """Decorator for pre_execute function."""
        self._pre_execute = func
        return func

    def post_execute(self, func):
        """Decorator for post_execute function."""
        self._post_execute = func
        return func
