import random
from typing import List, Tuple
from black.lines import Callable


# Функция приспособленности
def fitness(func: Callable[[float, float], float], ind: Tuple[float, float]) -> float:
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
    func: Callable[[float, float], float],
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
    func: Callable[[float, float], float],
    parent1: Tuple[float, float],
    parent2: Tuple[float, float],
) -> Tuple[float, float]:
    if random.random() < 0.5:
        return (parent1[0] + parent2[0]) / 2, (parent1[1] + parent2[1]) / 2
    else:
        return parent1 if fitness(func, parent1) < fitness(func, parent2) else parent2


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
            child1, child2 = crossover(func, parent1, parent2), crossover(
                func, parent1, parent2
            )
            new_population.extend(
                [
                    mutate(child1, mutation_rate, x_range, y_range),
                    mutate(child2, mutation_rate, x_range, y_range),
                ]
            )
        population = new_population
        best_in_population: Tuple[float, float] = min(
            population, key=lambda x: fitness(func, x)
        )
        if fitness(func, best_in_population) < fitness(func, best_solution):
            best_solution = best_in_population

    return best_solution


class GeneticAlgorithm:
    def __init__(self):
        pass

    def __call__(
        self,
        func: Callable[[float, float], float],
        x_range: Tuple[float, float] = (-5, 5),
        y_range: Tuple[float, float] = (-5, 5),
        pop_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
    ) -> Tuple[float, float]:
        self.func = func
        population: List[Tuple[float, float]] = self.initial_population(
            pop_size, x_range, y_range
        )
        best_solution: Tuple[float, float] = min(population, key=self.fitness)

        for _ in range(generations):
            new_population = []
            for _ in range(pop_size // 2):
                parent1, parent2 = self.select_parents(population)
                child1, child2 = self.crossover(parent1, parent2), self.crossover(
                    parent1, parent2
                )
                new_population.extend(
                    [
                        self.mutate(child1, mutation_rate, x_range, y_range),
                        self.mutate(child2, mutation_rate, x_range, y_range),
                    ]
                )
            population = new_population
            best_in_population: Tuple[float, float] = min(population, key=self.fitness)
            if self.fitness(best_in_population) < self.fitness(best_solution):
                best_solution = best_in_population

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

    # Селекция родителей (приспособленность как веса)
    def select_parents(
        self,
        population: List[Tuple[float, float]],
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        fitness_values: List[float] = [self.fitness(ind) for ind in population]
        parents: List[Tuple[float, float]] = random.choices(
            population,
            weights=[1.0 / f for f in fitness_values],
            k=2,
        )
        return parents[0], parents[1]

    # Кроссовер (среднее арифметическое)
    def crossover(
        self,
        parent1: Tuple[float, float],
        parent2: Tuple[float, float],
    ) -> Tuple[float, float]:
        if random.random() < 0.5:
            return (parent1[0] + parent2[0]) / 2, (parent1[1] + parent2[1]) / 2
        else:
            return parent1 if self.fitness(parent1) < self.fitness(parent2) else parent2

    # Мутация (случайное изменение)
    def mutate(
        self,
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
