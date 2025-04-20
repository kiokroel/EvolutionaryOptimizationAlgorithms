import random
from copy import deepcopy


class Field:
    def __init__(
        self, size: int = 32, food_coords: list[tuple[int, int]] = None
    ):
        self.size = size
        food_coords = food_coords or []
        self.grid = self.initialize_grid(food_coords)

    def initialize_grid(self, food_coords):
        grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        for x, y in food_coords:
            grid[y][x] = 1
        return grid

    def count_food(self):
        return sum(sum(row) for row in self.grid)


class AntSimulator:
    def __init__(self, grid: list[list[int]]):
        self.grid = grid
        self.path: list[tuple[int, int]] = [(0, 0)]
        self.pos = (0, 0)
        self.dir = 1  # 0-up, 1-right, 2-down, 3-left
        self.score = 0
        self.steps = 0

    def sense_food_ahead(self) -> int:
        x, y = self.pos
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.dir]
        nx, ny = x + dx, y + dy
        return int(
            0 <= nx < len(self.grid[0])
            and 0 <= ny < len(self.grid)
            and self.grid[ny][nx]
        )

    def turn_left(self):
        self.dir = (self.dir - 1) % 4
        self.steps += 1

    def turn_right(self):
        self.dir = (self.dir + 1) % 4
        self.steps += 1

    def move_forward(self):
        x, y = self.pos
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.dir]
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(self.grid[0]) and 0 <= ny < len(self.grid):
            self.pos = (nx, ny)
            self.path.append(self.pos)
            if self.grid[ny][nx]:
                self.score += 1
                self.grid[ny][nx] = 0
        self.steps += 1


class FiniteStateMachine:
    def __init__(self, num_states=8):
        self.num_states = num_states
        self.transitions = []
        for _ in range(num_states):
            state = []
            for _ in range(2):  # 0 - нет еды, 1 - есть еда
                action = random.randint(0, 2)  # 0 - left, 1 - right, 2 - move
                next_state = random.randint(0, num_states - 1)
                state.append((action, next_state))
            self.transitions.append(state)

    def get_action(self, current_state, sensor_input):
        return self.transitions[current_state][sensor_input]


def fitness(
    fsm, simulator, max_steps=900
) -> tuple[FiniteStateMachine, int, list[tuple[int, int]]]:
    current_state = 0
    while simulator.steps < max_steps:
        sensor_input = simulator.sense_food_ahead()
        action, next_state = fsm.get_action(current_state, sensor_input)
        if action == 0:
            simulator.turn_left()
        elif action == 1:
            simulator.turn_right()
        else:
            simulator.move_forward()
        current_state = next_state
    return fsm, simulator.score, simulator.path


# Генерация начальной популяции
def initial_population(
    population_size: int, num_states: int
) -> list[FiniteStateMachine]:
    return [FiniteStateMachine(num_states) for _ in range(population_size)]


# Селекция родителей (приспособленность как веса)
def select_parents(
    population: list[FiniteStateMachine], fitness_values
) -> tuple[FiniteStateMachine, FiniteStateMachine]:
    parents: list[FiniteStateMachine] = random.choices(
        population,
        weights=[f[1] or 1 for f in fitness_values],
        k=2,
    )
    return parents[0], parents[1]


# Кроссовер (среднее арифметическое)
def crossover(
    parent1: FiniteStateMachine, parent2: FiniteStateMachine
) -> FiniteStateMachine:
    child = deepcopy(parent1)
    for i in range(child.num_states):
        if random.random() < 0.5:
            child.transitions[i] = deepcopy(parent2.transitions[i])
    return child


# Мутация (случайное изменение)
def mutate(ind: FiniteStateMachine, mutation_rate: float) -> FiniteStateMachine:
    mutated = deepcopy(ind)
    for i in range(mutated.num_states):
        for j in range(2):
            if random.random() < mutation_rate:
                mutated.transitions[i][j] = (
                    random.randint(0, 2),
                    random.randint(0, mutated.num_states - 1),
                )
    return mutated


def genetic_algorithm(
    grid: list[list[int]],
    pop_size: int = 50,
    generations: int = 100,
    num_states: int = 8,
    mutation_rate: float = 0.05,
    steps: int = 900
):
    population: list[FiniteStateMachine] = initial_population(
        pop_size, num_states
    )
    pop_fitness = [fitness(fsm, AntSimulator(deepcopy(grid)), steps) for fsm in population]
    best_solution, best_score, best_path = max(pop_fitness, key=lambda x: x[1])

    for _ in range(generations):
        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = select_parents(population, pop_fitness)
            child1, child2 = crossover(parent1, parent2), crossover(
                parent1, parent2
            )
            new_population.extend(
                [
                    mutate(child1, mutation_rate),
                    mutate(child2, mutation_rate),
                ]
            )
        population = new_population

        pop_fitness = [fitness(fsm, AntSimulator(deepcopy(grid)), steps) for fsm in population]
        best_in_population, best_pop_score, best_pop_path = max(pop_fitness,
                                                   key=lambda x: x[1])

        if best_pop_score > best_score:
            best_solution = best_in_population
            best_score = best_pop_score
            best_path = best_pop_path

    return best_solution, best_score, best_path

size = 32

food_coords = [
    (0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0),(9,0),(10,0),(11,0),(12,0),(13,0),(14,0),
    (14,1),(14,2),(14,3),(14,4),(14,5),(14,6),(14,7),(14,8),(14,9),(14,10),(14,11),(14,12),
    (13,12),(12,12),(11,12),(10,12),(9,12),(8,12),(7,12),(6,12),(5,12),(4,12),(3,12),(2,12),(1,12),
    (1,13),(1,14),(1,15),(1,16),(1,17),(1,18),(1,19),(1,20),(1,21),(1,22),(1,23),(1,24),
    (2,24),(3,24),(4,24),(5,24),(6,24),(7,24),(8,24),(9,24),(10,24),(11,24),(12,24),(13,24),
    (13,25),(13,26),(13,27),(13,28),(13,29),
    (12,29),(11,29),(10,29),(9,29),(8,29),(7,29),(6,29),(5,29),(4,29),(3,29),(2,29),(1,29),(0,29)
]


def generate_john_muir_trail(size=32):
    food_coords = []
    x, y = 0, size // 2

    for _ in range(size * 2):
        dx = random.choice([0, 0, 1, -1])
        dy = random.choice([0, 0, 1, -1])

        x = (x + dx) % size
        y = (y + dy) % size

        food_coords.append((x, y))

    return Field(size=size, food_coords=food_coords)


field = generate_john_muir_trail()

# Генерация случайных уникальных координат

# total_food = 89
#
# # Создаем все возможные координаты
# all_coords = [(x, y) for x in range(size) for y in range(size)]
# # Выбираем случайные 89 без повторений
# import random
#
# random.seed(42) # Для воспроизводимости
# food_coords = random.sample(all_coords, total_food)


#field = Field(size=size, food_coords=food_coords)

total_food = field.count_food()


best_fsm, best_score, path = genetic_algorithm(field.grid, pop_size=50, generations=500, mutation_rate=0.1, steps=900)

print(f"Найдено еды: {best_score}")
print(f"Исходная еда: {field.count_food()}")
print(f"Пройдено шагов: {len(path)}")
print(f"Пример пути: {path}")

def print_path(path, field):
    grid = field.grid.copy()
    for y in range(32):
        for x in range(32):
            if field.grid[y][x]:
                grid[y][x] = 'F'
            else:
                grid[y][x] = '.'

    for x, y in path:
        if grid[y][x] == 'F':
            grid[y][x] = 'X'
        else:
            grid[y][x] = '*'

    for row in grid:
        print(' '.join(row[:32]))

print(f"\nBest fitness: {best_score} ({best_score / total_food * 100:.1f}%)")
print("\nBest path visualization:")
print_path(path, field)



def action_to_str(action_code: int) -> str:
    return {
        0: "Turn Left",
        1: "Turn Right",
        2: "Move Forward"
    }.get(action_code, "Unknown Action")

print("\nBest FSM structure:")
for state_idx, transitions in enumerate(best_fsm.transitions):
    print(f"State {state_idx}:")

    # Для ситуации с едой (sensor_input=1)
    food_action, food_next = transitions[1]
    print(f"  Food detected:")
    print(f"    Action: {action_to_str(food_action)}")
    print(f"    Next state: {food_next}")

    # Для ситуации без еды (sensor_input=0)
    no_food_action, no_food_next = transitions[0]
    print(f"  No food detected:")
    print(f"    Action: {action_to_str(no_food_action)}")
    print(f"    Next state: {no_food_next}")
    print("-" * 30)

