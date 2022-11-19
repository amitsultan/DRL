import gym
import tensorflow as tf
from gym import Env
import numpy as np
from tensorflow.keras.models import clone_model
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential


def get_ann(input_dim, output_dim):
    model = Sequential(
        [
            Dense(input_dim, activation="relu", name="input_layer"),
            Dense(input_dim * 4, activation="relu", name="hidden_1"),
            Dense(input_dim * 8, activation="relu", name="hidden_2"),
            Dense(input_dim * 4, activation="relu", name="hidden_3"),
            Dense(output_dim, activation='linear', name='output_layer')
        ])
    model.compile(loss='mse', optimizer='adam')
    return model

class dqn:

    def __init__(self, env, experience_cap, batch_size, C=3, epsilon=1.0):
        self.env = env
        self.observation_space_size = env.observation_space.shape[0]
        self.action_space_size = env.action_space.n
        self.epsilon = epsilon
        self.experience_lst = []
        self.experience_cap = experience_cap
        self.bath_size = batch_size
        self.ann = get_ann(input_dim=self.observation_space_size, output_dim=self.action_space_size)
        self.target_network = get_ann(input_dim=self.observation_space_size, output_dim=self.action_space_size)
        self.gamma = 0.97  # discount
        self.C = C
        self.decaying_rate = 0.9995

    def train(self, episodes, n_steps):
        total_steps = 0
        for episodes in tqdm(range(episodes)):
            state = self.env.reset()[0]
            rewards = []
            for step in range(n_steps):
                total_steps += 1
                state, action, reward, new_state, is_done, info = self.perform_action(state)
                rewards.append(reward)
                experience_slot = state, action, reward, new_state, is_done, info
                self.insert_memory(experience_slot)
                X_train, y_train = self.get_batch(self.bath_size)
                self.ann.fit(x=X_train, y=y_train, verbose=0)
                if is_done:
                    break
            self.epsilon = max(self.epsilon * self.decaying_rate, 0.002)
            print(np.array(rewards).sum())
            if (episodes + 1) % self.C == 0:
                print('updated target weights')
                self.target_network = self.clone_model()


    def preprocess_batch(self, batch):
        # batch contains many <state, action, reward, new_state, is_done, info>
        states = [state[0] for state in batch]
        actions = [state[1] for state in batch]
        rewards = [state[2] for state in batch]


    def insert_memory(self, memory_item):
        if len(self.experience_lst) == self.experience_cap:
            self.experience_lst.pop(0)
        self.experience_lst.append(memory_item)

    def perform_action(self, state):
        e_rand = np.random.random()
        predictions = self.ann.predict(state.reshape(1, -1))
        if e_rand < self.epsilon:  # exploration
            action = np.random.randint(self.action_space_size)  # random action
        else:  # exploitation
            action = np.argmax(predictions)
        new_state, reward, terminated, truncated, info = self.env.step(action)
        is_done = False
        if terminated or truncated:
            is_done = True
            new_state = self.env.reset()[0]
        return state, action, reward, new_state, is_done, info

    def get_batch(self, batch_size):
        memory = np.array(self.experience_lst)
        index = np.random.choice(memory.shape[0], batch_size, replace=True)
        learning_batch = memory[index, :]
        X, y = [], []
        states = [row[0] for row in learning_batch]
        actions = [row[1] for row in learning_batch]
        rewards = [row[2] for row in learning_batch]
        new_states = [row[3] for row in learning_batch]
        new_states = np.array(new_states)
        X = np.array(states)
        target = self.ann.predict(X)
        all_t = self.target_network.predict(new_states)
        for i in range(len(target)):
            is_done = learning_batch[i][4]
            if is_done:
                target[0][actions[i]] = rewards[i]
            else:
                t = all_t[i]
                target[0][actions[i]] = rewards[i] + self.gamma * np.amax(t)
        y.append(target)
        y = np.array(y)
        return X, y[0]

    def clone_model(self):
        model_copy = tf.keras.models.clone_model(self.ann)
        model_copy.build((None, self.observation_space_size))  # replace 10 with number of variables in input layer
        model_copy.compile(optimizer='adam', loss='mse')
        model_copy.set_weights(self.ann.get_weights())
        return model_copy

env = gym.make('CartPole-v1')
d = dqn(env, experience_cap=10000, batch_size=512, epsilon=1.0)
d.train(100, 100)