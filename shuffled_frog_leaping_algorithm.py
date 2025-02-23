import random
from typing import Callable, Tuple, List

import numpy as np


def ackley(x: float, y: float) -> float:
    return (
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        + np.e
        + 20
    )


def rastrigin(x, y):
    return (
        20
        + (x**2 - 10 * np.cos(2 * np.pi * x))
        + (y**2 - 10 * np.cos(2 * np.pi * y))
    )


def holder_table(x, y):
    return -np.abs(
        np.sin(x)
        * np.cos(y)
        * np.exp(np.abs(1 - np.sqrt(x**2 + y**2) / np.pi))
    )


class ShuffledFrogLeaping:
    def __init__(self):
        pass

    def __call__(
        self,
        func: Callable[[float, float], float],
        x_range: Tuple[float, float] = (-5, 5),
        y_range: Tuple[float, float] = (-5, 5),
        population_size: int = 50,
        num_memeplexes: int = 5,
        num_iterations: int = 100,
    ) -> Tuple[float, float]:
        self.func = func
        self.x_range = x_range
        self.y_range = y_range

        # Инициализация популяции
        population = self.initial_population(population_size, x_range, y_range)

        # Основной цикл
        for iteration in range(num_iterations):
            # Сортировка популяции по приспособленности
            population.sort(key=self.fitness)

            # Разделение на мемеплексы
            memeplexes = self.split_into_memeplexes(population, num_memeplexes)

            # Локальный поиск в каждом мемеплексе
            for i in range(len(memeplexes)):
                memeplexes[i] = self.local_search(memeplexes[i])

            # Объединение мемеплексов
            population = [frog for memeplex in memeplexes for frog in memeplex]

        # Возвращаем лучшее решение
        best_solution = min(population, key=self.fitness)
        return best_solution

    # Функция приспособленности
    def fitness(self, ind: Tuple[float, float]) -> float:
        return self.func(*ind)

    # Генерация начальной популяции
    def initial_population(
        self,
        population_size: int,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        return [
            (
                random.uniform(x_range[0], x_range[1]),
                random.uniform(y_range[0], y_range[1]),
            )
            for _ in range(population_size)
        ]

    # Разделение популяции на мемеплексы
    def split_into_memeplexes(
        self,
        population: List[Tuple[float, float]],
        num_memeplexes: int,
    ) -> List[List[Tuple[float, float]]]:
        memeplexes = [[] for _ in range(num_memeplexes)]
        for i, ind in enumerate(population):
            memeplexes[i % num_memeplexes].append(ind)
        return memeplexes

    # Локальный поиск в мемеплексе
    def local_search(
        self,
        memeplex: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        # Сортировка по приспособленности
        memeplex.sort(key=self.fitness)
        best_frog = memeplex[0]
        worst_frog = memeplex[-1]

        new_frog = (
            worst_frog[0]
            + random.uniform(0, 1) * (best_frog[0] - worst_frog[0]),
            worst_frog[1]
            + random.uniform(0, 1) * (best_frog[1] - worst_frog[1]),
        )

        # Ограничение в пределах bounds
        new_frog = (
            float(np.clip(new_frog[0], self.x_range[0], self.x_range[1])),
            float(np.clip(new_frog[1], self.y_range[0], self.y_range[1])),
        )

        # Замена худшей лягушки, если новое решение лучше
        if self.fitness(new_frog) < self.fitness(worst_frog):
            memeplex[-1] = new_frog

        return memeplex


X_RANGE = (-10, 10)
Y_RANGE = (-10, 10)
NUM_FROGS = 200
NUM_MEMEPLEXES = 20
NUM_ITERATIONS = 1000

alg = ShuffledFrogLeaping()


best_solution = alg(
    func=ackley,
    x_range=X_RANGE,
    y_range=Y_RANGE,
    population_size=NUM_FROGS,
    num_memeplexes=NUM_MEMEPLEXES,
    num_iterations=NUM_ITERATIONS,
)
print(
    f"Лучшее решение: {best_solution}, Значение функции Экли: {ackley(*best_solution)}"
)

# Функция Растригина
best_solution = alg(
    func=rastrigin,
    x_range=X_RANGE,
    y_range=Y_RANGE,
    population_size=NUM_FROGS,
    num_memeplexes=NUM_MEMEPLEXES,
    num_iterations=NUM_ITERATIONS,
)
print(
    f"Лучшее решение: {best_solution}, Значение функции Растригина: {rastrigin(*best_solution)}"
)

# Табличная функция Хольдера
best_solution = alg(
    func=holder_table,
    x_range=X_RANGE,
    y_range=Y_RANGE,
    population_size=NUM_FROGS,
    num_memeplexes=NUM_MEMEPLEXES,
    num_iterations=NUM_ITERATIONS,
)
print(
    f"Лучшее решение: {best_solution}, Значение функции Хольдера: {holder_table(*best_solution)}"
)
