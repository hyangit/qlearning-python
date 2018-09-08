from q_state import next_state, random_state, actions
from q_learning import QLearning
from q_table import QTable

if __name__ == "__main__":
    episode = 100
    model_save_interval = 10

    table = QTable(actions)
    learning = QLearning(table)

    for step in range(episode):
        init_state = random_state()
        i = 0
        reward = 0
        while reward != 1:
            state = init_state
            while True:
                i += 1
                action = learning.choose_action(state)
                state2, reward, done = next_state(state, action, table)
                learning.learn(state, action, reward, state2, done)
                if done:
                    break
                state = state2
        print(init_state, i, len(table.q_table))
        if (step + 1) % model_save_interval == 0:
            table.save()
