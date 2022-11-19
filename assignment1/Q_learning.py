import numpy as np
from gym import Env
from tqdm import tqdm


class q_learning:

    def __init__(self, env: Env, learning_rate=0.5, discount_factor=0.9, decaying_rate=0.001, epsilon=1.0):
        self.env = env
        self.steps_to_solution_history = []
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.decaying_rate = decaying_rate
        self.epsilon = epsilon
        self.episodes_rewards = []
        self.lr_history = [self.lr]
        self.q_table_history = {}
        print(f"####################\n"
              f"Hyper-params: \n"
              f"Learning-Rate: {learning_rate}\n"
              f"Discount factor: {discount_factor}\n"
              f"Decaying-Rate: {decaying_rate}\n"
              f"Epsilon: {epsilon}\n"
              f"####################\n")

    def run_experiment(self, episodes, steps_per_episode):
        self.steps_to_solution_history = []
        steps_average = []
        for episode in tqdm(range(episodes + 1)):
            if episode % 100 == 0:
                self.steps_to_solution_history.append(np.array(steps_average).mean())
                steps_average = []
            if episode % 500 == 0:
                self.q_table_history[episode] = np.copy(self.q_table)
            state = self.env.reset()
            rewards = []
            steps_to_solution = 0
            for step in range(steps_per_episode):
                e_rand = np.random.random()
                if e_rand < self.epsilon:  # exploration
                    action = self.env.action_space.sample()
                else:  # exploitation
                    action = np.argmax(self.q_table[state[0]])
                new_state, reward, terminated, truncated, info = self.env.step(action)
                if terminated or truncated:
                    new_state = self.env.reset()[0]
                    steps_average.append(steps_to_solution + 1)
                    steps_to_solution = 0
                else:
                    steps_to_solution += 1
                self.q_table[state[0], action] = self.q_table[state[0], action] + \
                                        self.lr * (reward + self.discount_factor * np.max(self.q_table[new_state]) - self.q_table[state[0], action])

                rewards.append(reward)
                # self.q_table[state, action] = (1 - self.lr) * self.q_table[state, action] + self.lr * target
                state = (new_state, info)
            # Update epsilon
            if steps_to_solution != 0:
                steps_average.append(100)
            self.epsilon = max(self.epsilon - self.decaying_rate, 0)
            self.episodes_rewards.append(np.sum(rewards))

        print('done')