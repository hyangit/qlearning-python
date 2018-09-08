import numpy as np
import pandas as pd
import os.path


class QTable:
    def __init__(self, actions, model_file="model.pkl"):
        self.actions = actions
        self.model_file = model_file
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.load()

    def load(self):
        if os.path.exists(self.model_file):
            self.q_table = pd.read_pickle(self.model_file)

    def save(self):
        self.q_table.to_pickle(self.model_file)

    def exists(self, state):
        return state in self.q_table.index

    def append_if_not_exists(self, state):
        if not self.exists(state):
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def get_index_of_max_value(self, state):
        state_action = self.q_table.loc[state, :]
        return np.random.choice(state_action[state_action == np.max(state_action)].index)

    def set(self, state, action, value):
        self.q_table.loc[state, action] = value

    def get(self, state, action):
        return self.q_table.loc[state, action]

    def get_max_value(self, state):
        return self.q_table.loc[state, :].max()

    def is_all_zero(self, state):
        state_action = self.q_table.loc[state, :]
        return np.min(state_action) == np.max(state_action) == 0
