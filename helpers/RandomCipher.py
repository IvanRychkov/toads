import numpy as np


class RandomCipher:
    """Объект-шифровщик.
    Преобразует данные, умножая их на случайную обратимую матрицу. Умеет восстанавливать данные """

    def __init__(self, random_state=None):
        """random_state: фиксирует случайное состояние для случайной трансформации."""
        self.__rstate = random_state

    def fit(self, X):
        """Принимает матрицу и создаёт внутри себя матрицу-ключ."""
        self.__make_matrix(X)
        return self

    def transform(self, X):
        """Преобразует матрицу ключом."""
        return X @ self.__P

    def inverse_transform(self, X):
        """Умножает матрицу на обратную матрицу-ключ."""
        return X @ self.__P_inv

    def fit_transform(self, X):
        """Принимает на вход матрицу и преобразует её случайным образом."""
        self.fit(X)
        return self.transform(X)

    def __make_matrix(self, X):
        """Создаёт случайную матрицу.
        Пересоздаёт, если матрица необратима."""
        # Задаём случайное состояние, если оно указано при создании объекта
        np.random.seed(self.__rstate)

        # Создаём квадратную матрицу заданного размера
        self.__P = np.random.randint(1000, size=(X.shape[1], X.shape[1]))

        # С осторожностью находим обратную матрицу
        try:
            self.__P_inv = np.linalg.inv(self.__P)

        # Если нет обратной, повторяем
        except np.linalg.LinAlgError:
            self.__make_matrix(X)
