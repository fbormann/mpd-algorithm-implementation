import copy
from typing import List, Tuple

import numpy as np


def get_possible_states(state: (int, int), max_rows: int, max_columns: int, goal_states: List[Tuple[int, int]]) -> List[
    Tuple[int, int]]:
    possible_states = []

    if state in goal_states:
        return possible_states # return possible state empty for goal states

    if state[0] != 0:
        # it can go up
        possible_states.append((state[0] - 1, state[1]))

    if state[1] != 0:
        # it can go left
        possible_states.append((state[0], state[1] - 1))

    if state[0] != max_rows - 1:
        # it can go down
        possible_states.append((state[0] + 1, state[1]))
    if state[1] != max_columns - 1:
        # it can go right
        possible_states.append((state[0], state[1] + 1))

    # index: int = 0
    # while index < len(possible_states) - 1:
    #     if possible_states[index] in goal_states:
    #         del possible_states[index]
    #     index += 1
    return possible_states


def update_utility_matrix(rw: np.array, value: np.array) -> np.array:
    pass


def update_calculations(policies, rw, value: np.array, goal_states: List[Tuple[int, int]]):
    max_rows = value.shape[0]
    max_columns = value.shape[1]

    new_value: np.array = copy.deepcopy(value)

    i: int = 0
    while i < max_rows:
        j: int = 0
        while j < max_columns:
            possible_states: List[(int, int)] = get_possible_states((i, j), max_rows, max_columns, goal_states)
            state_scores = list(map(lambda state: value[state[0], state[1]], possible_states))
            if len(state_scores) > 0:
                new_value[i, j] = rw[i, j] + state_scores[np.argmax(state_scores)]
            else:
                new_value[i, j] = rw[i, j]
            j += 1
        i += 1

    # for every position, calculate the utility for that position
    i: int = 0
    while i < max_rows:
        j: int = 0
        while j < max_columns:
            possible_states: List[(int, int)] = get_possible_states((i, j), max_rows, max_columns, goal_states)
            j += 1
        i += 1
    # calculate the policy given the utility for that

    return policies, new_value


def run_experiments(default_reward: float):
    goal_states: List[Tuple[int, int]] = [(0, 3), (1, 3)]
    # define initial utilities matrix
    matrix: np.array = np.zeros((3, 4))

    print(matrix)

    # define reward matrix

    reward_matrix: np.array = np.zeros((3, 4))
    reward_matrix.fill(default_reward)
    reward_matrix[0, 3] = 1
    reward_matrix[1, 3] = -1
    reward_matrix[2, 3] = 0.2
    reward_matrix[1, 1] = -0.5

    print(reward_matrix)

    # define utility matrix
    utility_matrix: np.array = np.zeros((3, 4))

    # define policies
    policies = [[], [], [], []]

    #  a minha equação de estabilidade pode ser entendida da seguinte maneira
    # se a mudança de valores não modificou a soma total em mais de 10%, então a matriz está equilibrada.
    previous_utility_matrix = copy.deepcopy(utility_matrix)
    policies, utility_matrix = update_calculations(policies, reward_matrix, utility_matrix, goal_states)
    times = 10
    i = 0
    while i < times:
        previous_utility_matrix = copy.deepcopy(utility_matrix)
        policies, utility_matrix = update_calculations(policies, reward_matrix, previous_utility_matrix, goal_states)
        i += 1

    print("stabilized matrix")
    # utility_matrix: np.array = update_utility_matrix(reward_matrix, utility_matrix)

    # define policies given utility matrix


if __name__ == '__main__':
    initial_state = (0, 0)  # change it to be a input of a given user.

    possible_rewards = [-0.4, -0.04, -0.0004]
    fading_factor = [0.8, 0.3, 0.1, 0.05, 0.0005]
    for chosen_reward in possible_rewards:
        run_experiments(chosen_reward)