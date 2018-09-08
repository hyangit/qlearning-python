import numpy as np
import math

finished_state = "123456780"
actions = [0, 1, 2, 3]

state_len = len(finished_state)
row_size = int(math.sqrt(state_len))


def random_state(difficulty=10):
    state = finished_state
    for i in range(difficulty):
        action = np.random.choice(actions)
        state, _, _ = next_state(state, action, None)
    return state


def next_state(state, action, q_table):
    index = -1
    index0 = state.index("0")
    if action == 0:
        index = index0 - row_size
    elif action == 1:
        index = index0 + row_size
    elif action == 2:
        index = index0 - 1
    elif action == 3:
        index = index0 + 1

    reward = 0
    done = False
    if 0 <= index < state_len:
        state = swap(state, index0, index)
    else:
        reward = -1
        done = True
        return state, reward, done

    if state == finished_state:
        reward = 1
        done = True
    elif q_table is not None and is_go_back(state, q_table):
        reward = -0.1
    return state, reward, done


def swap(state, i, j):
    m = min([i, j])
    n = max([i, j])
    return state[:m] + state[n] + state[m + 1:n] + state[m] + state[n + 1:]


def is_go_back(state, q_table):
    return not (q_table.exists(state) and q_table.is_all_zero(state))
