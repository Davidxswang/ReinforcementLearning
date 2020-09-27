import torch
import gym
from utils import draw, gen_epsilon_greedy_policy
from cliff_walking_q_learning import q_learning
from windy_gridworld_SARSA import sarsa




if __name__ == '__main__':
    env = gym.make('Taxi-v3')
    n_episode = 500
    gamma = 1
    alpha_options = [0.4, 0.5, 0.6]
    epsilon_options = [0.1, 0.03, 0.01]
    
    for alpha in alpha_options:
        for epsilon in epsilon_options:
            gen_policy = gen_epsilon_greedy_policy(env.action_space.n, epsilon)
            _, _, lengths, rewards = sarsa(env, gamma, n_episode, alpha, gen_policy)
            reward_per_step = [reward/length for reward, length in zip(rewards, lengths)]
            print(f"{alpha=} {epsilon=}")
            print(f"average length per episode: {sum(lengths) / n_episode}")
            print(f"average reward per episode: {sum(rewards) / n_episode}")
            print(f"average reward per step: {sum(reward_per_step) / n_episode}")