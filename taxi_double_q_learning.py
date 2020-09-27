import torch
import gym
from utils import draw, gen_epsilon_greedy_policy
from cliff_walking_q_learning import q_learning
from windy_gridworld_SARSA import sarsa




if __name__ == '__main__':
    env = gym.make('Taxi-v3')
    n_episode = 1000
    gamma = 1
    alpha = 0.4
    epsilon = 0.1
    gen_policy = gen_epsilon_greedy_policy(env.action_space.n, epsilon)
    Q_q_learning, policy_q_learning, lengths_q_learning, rewards_q_learning = q_learning(env, gamma, n_episode, alpha, gen_policy)
    draw('taxi_q_learning_lengths.png', 'Lengths over time using q_learning', 'Episode', 'Length', lengths_q_learning)
    draw('taxi_q_learning_rewards.png', 'Rewards over time using q_learning', 'Episode', 'Length', rewards_q_learning)
    Q_sarsa, policy_sarsa, lengths_sarsa, rewards_sarsa = sarsa(env, gamma, n_episode, alpha, gen_policy)
    draw('taxi_sarsa_lengths.png', 'Lengths over time using sarsa', 'Episode', 'Length', lengths_sarsa)
    draw('taxi_sarsa_rewards.png', 'Rewards over time using sarsa', 'Episode', 'Length', rewards_sarsa)