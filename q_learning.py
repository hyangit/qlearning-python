import numpy as np
from q_state import actions


class QLearning:
    def __init__(self, q_table, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.q_table = q_table
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.e_greedy = e_greedy

    def choose_action(self, state):
        self.q_table.append_if_not_exists(state)
        if np.random.uniform() < self.e_greedy:
            action = self.q_table.get_index_of_max_value(state)
        else:
            action = np.random.choice(actions)
        return action

    def learn(self, state, action, reward, state2, done):
        self.q_table.append_if_not_exists(state2)
        value = self.q_table.get(state, action)
        if done:
            next_state_max_value = 0
        else:
            next_state_max_value = self.q_table.get_max_value(state2)
        value += self.learning_rate * (reward + self.reward_decay * next_state_max_value - value)
        self.q_table.set(state, action, value)
