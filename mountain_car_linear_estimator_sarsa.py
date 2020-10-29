import gym
import torch
from linear_estimator import LinearEstimator
from utils import gen_epsilon_greedy_policy, draw


def sarsa(env, estimator, n_episode, n_action, gamma=1.0, epsilon=0.1, epsilon_decay=0.99):
    r"""
    SARSA algorithm using Function Approximation

    Args:
        env: OpenAI environment
        estimator: the estimator
        n_episode: number of episode
        n_action: number of action
        gamma: the discount factor
        epsilon: controls the trade-off between exploitation and exploration
        epsilon_decay: decrease the epsilon over time
    Returns:
        list: total reward over episode
    """
    total_reward_episode = [0.0] * n_episode

    for episode in range(n_episode):
        policy = gen_epsilon_greedy_policy(n_action, epsilon * epsilon_decay ** episode, estimator)
        state = env.reset()
        is_done = False
        action = policy(state)

        while not is_done:
            next_state, reward, is_done, info = env.step(action)
            q_values_next = estimator.predict(next_state)
            next_action = policy(next_state)
            target = reward + gamma * q_values_next[next_action]
            estimator.update(state, action, target)
            total_reward_episode[episode] += reward

            if is_done:
                break
            state = next_state
            action = next_action
        
    return total_reward_episode


if __name__ == '__main__':
    env = gym.envs.make('MountainCar-v0')

    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    n_feature = 200
    lr = 0.03
    estimator = LinearEstimator(n_feature, n_state, n_action, lr)
    n_episode = 300
    total_reward_episode = sarsa(env, estimator, n_episode, n_action, gamma=1.0, epsilon=0.1, epsilon_decay=0.99)

    draw('mountain_car_linear_estimator_sarsa.png', 'Reward over episodes', 'episode', 'reward', total_reward_episode)