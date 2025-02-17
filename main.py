import random
from typing import List, Tuple
import numpy as np
from black.lines import Callable


# Функция Экли
def ackley(x: float, y: float) -> float:
    return (
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        + np.e
        + 20
    )


# Функция приспособленности
def fitness(func: Callable[[float,float], float] ,ind: Tuple[float, float]) -> float:
    return func(*ind)


# Генерация начальной популяции
def initial_population(
    population_size: int, x_range: Tuple[float, float], y_range: Tuple[float, float]
) -> List[Tuple[float, float]]:
    return [
        (random.uniform(x_range[0], x_range[1]), random.uniform(y_range[0], y_range[1]))
        for _ in range(population_size)
    ]


# Селекция родителей (приспособленность как веса)
def select_parents(
    func: Callable[[float,float], float],
    population: List[Tuple[float, float]],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    fitness_values: List[float] = [fitness(func, ind) for ind in population]
    parents: List[Tuple[float, float]] = random.choices(
        population,
        weights=[1.0 / f for f in fitness_values],
        k=2,
    )
    return parents[0], parents[1]


# Кроссовер (среднее арифметическое)
def crossover(
func: Callable[[float,float], float],
    parent1: Tuple[float, float], parent2: Tuple[float, float]
) -> Tuple[float, float]:
    if random.random() < 0.5:
        return (parent1[0] + parent2[0]) / 2, (parent1[1] + parent2[1]) / 2
    else:
        return parent1 if fitness(func,parent1) < fitness(func,parent2) else parent2


# Мутация (случайное изменение)
def mutate(
    solution: Tuple[float, float],
    mutation_rate: float,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
) -> Tuple[float, float]:
    if random.random() < mutation_rate:
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        return x, y
    return solution


def genetic_algorithm(
    func: Callable[[float, float], float],
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    pop_size: int = 50,
    generations: int = 100,
    mutation_rate: float = 0.1,
) -> Tuple[float, float]:
    population: List[Tuple[float, float]] = initial_population(
        pop_size, x_range, y_range
    )
    best_solution: Tuple[float, float] = min(population, key=lambda x: fitness(func, x))

    for _ in range(generations):
        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = select_parents(func, population)
            child1, child2 = crossover(func, parent1, parent2), crossover(func, parent1, parent2)
            new_population.extend(
                [
                    mutate(child1, mutation_rate, x_range, y_range),
                    mutate(child2, mutation_rate, x_range, y_range),
                ]
            )
        population = new_population
        best_in_population: Tuple[float, float] = min(population, key=lambda x: fitness(func, x))
        if fitness(func, best_in_population) < fitness(func, best_solution):
            best_solution = best_in_population

    return best_solution


def find_best_solution(func ,n, x_range, y_range, pop_size, generations, mutation_rate):
    best_solution = genetic_algorithm(
        func=func,
        x_range=x_range,
        y_range=y_range,
        pop_size=pop_size,
        generations=generations,
        mutation_rate=mutation_rate,
    )
    for _ in range(n - 1):
        solution = genetic_algorithm(
            func = func,
            x_range=x_range,
            y_range=y_range,
            pop_size=pop_size,
            generations=generations,
            mutation_rate=mutation_rate,
        )
        if abs(ackley(*solution)) < abs(ackley(*best_solution)):
            best_solution = solution
    return best_solution


N = 1
X_RANGE = (-5, 5)
Y_RANGE = (-5, 5)
POP_SIZE = 70
GENERATIONS = 300
MUTATION_RATE = 0.1
FUNC = ackley


best_solution = find_best_solution(
    func=FUNC,
    n=N,
    x_range=X_RANGE,
    y_range=Y_RANGE,
    pop_size=POP_SIZE,
    generations=GENERATIONS,
    mutation_rate=MUTATION_RATE,
)


print(
    f"Лучшее решение: {best_solution}, Значение функции Экли: {ackley(*best_solution)}"
)
