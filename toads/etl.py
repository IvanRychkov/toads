from tqdm import tqdm
from pydantic import BaseModel, ValidationError
from typing import Callable, Type, Iterable, List


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

        for batch in batch_iter(extracted, batch_size=self.batch_size):
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


def batch_iter(iterable, batch_size=10000):
    """Yields data in batches of arbitrary size."""
    buffer = []
    for i, v in enumerate(iterable, start=1):
        buffer.append(v)

        # If batch size reached
        if i % batch_size == 1:
            yield buffer
            buffer = []

    # If anything left in buffer
    if buffer:
        yield buffer


def execute_etl(extract_iterable: Iterable[dict],
                transform_model: Type[BaseModel] = None,
                load_func: Callable[[List[dict]], None] = None,
                pre_execute: Callable = None,
                post_execute: Callable = None,
                load_batch_size: int = 10000,
                skip_validation_errors: bool = False,
                verbose: bool = True):
    """Runs an ETL,
    extracting data from any iterable,
    transforming and validating each object with pydantic models,
    putting it into batches for loading with user defined function."""
    assert iter(extract_iterable), 'extract_iterable must be iterable'
    assert transform_model is None or issubclass(transform_model, BaseModel), 'transform_model must be either subclass ' \
                                                                              'of pydantic.main.BaseModel or None'
    assert callable(load_func), 'load_func must be callable'
    assert callable(pre_execute) or not post_execute, 'pre_execute must be callable'
    assert callable(post_execute) or not post_execute, 'post_execute must be callable'

    def load_batch(b: list):
        """Загрузка батча при помощи функции load_func"""
        if verbose:
            print(f'loading {len(b)} objects...')
        if len(b) != 0:
            load_func(b)

    def announce_batch(c: int):
        """Выводит сообщение об обработке очередного батча."""
        if verbose:
            print(f'extracting and validating batch #{c}')

    if pre_execute:
        if verbose:
            print('running pre-execute callback...')
        pre_execute()

    # Инициализируем переменные
    batch = []
    batch_counter = 0
    validation_error_count = 0
    next_load_at = load_batch_size  # Батч загружается на этой итерации

    # Запускаем ETL
    announce_batch(batch_counter)
    for i, e in enumerate(extract_iterable, start=1):
        # Валидация данных и добавление в батч на загрузку
        try:
            batch.append(transform_model(**e).dict() if transform_model else e)
        except ValidationError as e:
            # Если ошибка валидации
            if not skip_validation_errors:
                raise e
            validation_error_count += 1
            continue

        # Если достаточно данных, загружаем
        if i == next_load_at:
            load_batch(batch)
            batch.clear()  # очищаем батч
            assert len(batch) == 0, 'fatal error: batch is not empty!'
            batch_counter += 1
            next_load_at += load_batch_size  # Задаём следующую итерацию для загрузки
            announce_batch(batch_counter)
    # Загружаем оставшиеся данные
    if len(batch) > 0:
        load_batch(batch)

    if post_execute:
        if verbose:
            print('running post-execute callback...')
        post_execute()

    if verbose:
        print('done! invalid objects:', validation_error_count)


__all__ = ['ETL', 'batch_iter', 'execute_etl']
