import math
import torch
import gym
from utils import gen_epsilon_greedy_policy, draw
from linear_estimator import LinearEstimator


def q_learning(env, estimator, n_episode, n_action, gamma=1.0, epsilon=0.1, epsilon_decay=0.99):
    r"""
    Q-Learning algorithm using Function approximation

    Args:
        env: OpenAI environment
        estimator: the estimator to approximate the Q values
        n_episode: number of episode
        n_action: number of action
        gamma: discount factor
        epsilon: to control the trade-off between exploration and exploitation
        epsilon_decay: to control the epsilon over time
    Returns:
        list: total rewards for each of the episode
    """
    total_reward_episode = [0.0] * n_episode
    
    for episode in range(n_episode):
        policy = gen_epsilon_greedy_policy(n_action, epsilon * epsilon_decay ** episode, estimator)
        state = env.reset()
        is_done = False

        while not is_done:
            action = policy(state)
            next_state, reward, is_done, info = env.step(action)
            q_values_next = estimator.predict(next_state)
            target = reward + gamma * torch.max(q_values_next)
            estimator.update(state, action, target)
            total_reward_episode[episode] += reward

            if is_done:
                break
            state = next_state
        
    return total_reward_episode


if __name__ == "__main__":
    env = gym.envs.make("MountainCar-v0")

    n_action = env.action_space.n
    n_state = env.observation_space.shape[0]
    n_feature = 200
    lr = 0.03

    estimator = LinearEstimator(n_feature=n_feature, n_state=n_state, n_action=n_action, lr=lr)

    n_episode = 300
    
    total_reward_episode = q_learning(env, estimator, n_episode, n_action, gamma=1.0, epsilon=0.1, epsilon_decay=0.99)
    draw('mountain_car_linear_estimator_off_policy_q_learning.png', 'Reward over episodes', 'episode', 'reward', total_reward_episode)


