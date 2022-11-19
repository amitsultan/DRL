# import tensorflow as tf
import gym
from assignment1.Q_learning import *
import matplotlib.pyplot as plt

dataset = "FrozenLake-v1"


def frozen_lake_ql():
    env = gym.make(dataset, is_slippery=True)
    q_algo = q_learning(env=env)
    q_algo.run_experiment(5000, 100)
    fig, ax = plt.subplots()
    ax.plot(q_algo.steps_to_solution_history)
    ax.set_title('Average steps to reward - per 100 episodes')
    ax.set_ylabel('steps')
    ax.set_xlabel('Episode')
    plt.show()
    # plot reward
    fig, ax = plt.subplots()
    ax.plot(q_algo.episodes_rewards)
    ax.set_title('Reward values per episode')
    ax.set_ylabel('Reward')
    ax.set_xlabel('Episode')
    plt.show()
    # Plot q table in intervals
    fig, ax = plt.subplots(nrows=1, ncols=len(q_algo.q_table_history.keys()))
    plot_index = 0
    for index, table in q_algo.q_table_history.items():
        ax[plot_index].matshow(table, cmap='seismic')
        ax[plot_index].set_title(f'Episode: {index}')
        for (i, j), z in np.ndenumerate(table):
            ax[plot_index].text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        plot_index += 1
    plt.gca().set_xticks([x - 0.5 for x in plt.gca().get_xticks()][1:], minor='true')
    plt.gca().set_yticks([y - 0.5 for y in plt.gca().get_yticks()][1:], minor='true')
    plt.grid(which='minor')
    plt.show()


if __name__ == '__main__':
    frozen_lake_ql()
