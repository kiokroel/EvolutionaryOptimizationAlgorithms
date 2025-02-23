import numpy as np

from simple_gen_alg import GeneticAlgorithm


# Функция Экли
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


def find_best_solution(
    target_func, alg, n, x_range, y_range, pop_size, generations, mutation_rate
):
    best_solution = alg(
        func=target_func,
        x_range=x_range,
        y_range=y_range,
        pop_size=pop_size,
        generations=generations,
        mutation_rate=mutation_rate,
    )
    for _ in range(n - 1):
        solution = alg(
            func=target_func,
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
TARGET_FUNC = ackley
ALG = GeneticAlgorithm()


best_solution = find_best_solution(
    alg=ALG,
    target_func=TARGET_FUNC,
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
